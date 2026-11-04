/*
 * Greenskull's Depth Charge
 * A ReShade add-on that finds a game's depth buffer, measures it and places it
 * under the picture, then publishes it as ordinary ReShade depth.
 *
 * Author:  Dan Hammill (Greenskull)
 * Licence: MIT
 *
 * WHAT IT DOES
 *
 * ReShade's built-in "Generic Depth" add-on picks a depth-stencil buffer and leaves
 * the rest to you. Emulators are where that tends to fall apart, in three ways:
 *
 * 1. WHEN to capture. Generic Depth copies depth just before a clear, or at present.
 *    PCSX2's texture cache recycles targets instead of clearing them, so by present
 *    the contents can be gone. Depth Charge copies a target the moment a busy enough
 *    pass stops drawing into it (end of render pass, or depth-stencil unbind). It
 *    also catches depth kept in a colour target: DuckStation's PGXP depth lives in
 *    an R32_FLOAT render target, not a depth-stencil.
 *
 * 2. WHERE it lands. An emulator draws a fixed-aspect picture and fits it into the
 *    window, so a 4:3 or 16:9 picture sits between bars. Depth Charge finds the
 *    picture from those black bars (or you pick its shape) and places the depth
 *    over exactly that rectangle, at any internal resolution and window shape.
 *
 * 3. WHAT the values mean. Instead of guessing RESHADE_DEPTH_* values, Depth Finder
 *    measures the buffer over a few seconds of play and solves two numbers, a zero
 *    point and a far distance. With linearise=1 (the default) the add-on converts
 *    depth to real distance itself.
 *
 * The result is bound to the "DEPTH" semantic, so ordinary ReShade effects read it
 * with no changes on their side.
 *
 * SETUP
 *
 *   1. Install ReShade for your game or emulator, using the version "with full
 *      add-on support" from reshade.me. (The standard version cannot load add-ons.)
 *
 *   2. Make sure the game runs on Direct3D 10, 11 or 12. In an emulator, pick a
 *      Direct3D 11 or 12 renderer in its graphics settings. (Depth Charge needs one
 *      of these to measure and convert depth.)
 *
 *   3. Copy depth-charge.addon64 into the same folder as the game's .exe, where
 *      ReShade put its DLL. For a 32-bit game, use depth-charge.addon32 instead.
 *
 *   4. Turn off ReShade's own depth add-on. Start the game, press Home to open
 *      ReShade, go to the Add-ons tab, untick "Generic Depth", then restart the game.
 *      (Both bind DEPTH, and whichever runs last wins.)
 *
 *   5. DuckStation (PS1) only. DuckStation only makes a depth buffer when one of its own
 *      post-processing shaders asks for it, and DepthCharge_EnableDepth.fx is that ask.
 *      Copy it into DuckStation's resources\shaders\reshade\Shaders folder. Then in
 *      DuckStation: Settings > Post-Processing > Internal tab (not Display), turn on
 *      post-processing and add DepthCharge_EnableDepth. In Graphics > PGXP, turn on
 *      Geometry Correction and Depth Buffer. (The shader does not change the picture.)
 *
 *   ReShade's own depth settings ("Edit global preprocessor definitions") can stay as
 *   they are. Depth Charge writes its depth so that ReShade reads back the right
 *   distance, whatever they say.
 *
 *   Depth Charge saves its settings, and every game you measure, in depth-charge.cfg
 *   next to the add-on. Copy that file to keep your measurements or share them.
 *
 * ONLINE GAMES
 *   ReShade's add-on version can trigger anti-cheat in online games, which can
 *   lead to a ban. By installing Depth Charge you agree not to use it in
 *   competitive online games.
 *
 * CREDITS
 *   crosire        ReShade and its add-on API
 *   jlrouzies-fr   DLSS5-Feeder, the consumer this was first written to feed
 *   stenzek        DuckStation, and its PGXP depth buffer
 *   iCatButler     the first PS1 depth buffer built from PGXP
 *   Built with Claude (Anthropic).
 */

#include <imgui.h>
#include <reshade.hpp>
#include <d3dcompiler.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <map>
#include <string>
#include <cctype>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <atomic>
#include <cstdio>
#include <cstdarg>
#include <cmath>

using namespace reshade::api;

static uint32_t g_min_draws = 20;   // ignore UI, blits and shadow scratch passes

// One-shot diagnostic. Normally we only ever see the pass we chose, which is useless
// when the choice itself is wrong: a game whose scene target loses the draw-count race
// looks identical to a game with no depth at all. This dumps every candidate for a
// single frame - size, format, draws - so the selection can be judged instead of
// guessed at. Off unless the button is pressed; costs nothing while idle.
static int g_survey_frames = 0;
static int g_survey_binds  = 0;   // render-target binds logged during a survey

static void Log(const char *fmt, ...)
{
    char buf[512];
    va_list a; va_start(a, fmt);
    vsnprintf(buf, sizeof(buf), fmt, a);
    va_end(a);
    reshade::log::message(reshade::log::level::info, buf);
}


// depth-charge.cfg
//
// picture_aspect =
//     auto   find the picture from the black bars around it (default)
//     full   the picture fills the window (PCSX2 "Stretch", or 16:9 on a 16:9 screen)
//     4:3    the picture keeps 4:3 and is pillarboxed/letterboxed to fit
//     16:9   likewise for 16:9, or give any number such as 1.3333
//
// Re-read every few seconds, so you can change it without restarting the game.

static float       g_cfg_aspect = 0.0f;    // 0 = auto, -1 = full frame, else w/h
static bool        g_cfg_accumulate = true; // merge every pass that draws into the frame's depth target
// sparse_refresh (per game, Oct 2026): keep refreshing the depth target that already won even when its passes fall
// under the draw filter - a view of the sky can be a dozen draws, and without this the depth froze on the last busy
// frame (San Andreas: looking up at a tower, the effects kept the tower's old outline). Other targets still have to
// clear the full filter, so a tiny UI or scratch pass can never take over.
static bool        g_cfg_sparse_refresh = false;
// min_draws (per game, Oct 11 2026): the draw filter's floor for this game from its first frame. A PS1 frame in
// DuckStation is a 1-2 draw depth pass: with the default 20 nothing qualified until the automatic relaxation reached
// 1 draw after 900 frames (~30 s without depth - Spyro through a portal: no Minecraft blocks until then). 0 = default.
static uint32_t    g_cfg_min_draws = 0;

// linearisation
//
// A perspective depth buffer is  raw = A + B/distance.  A is what it asymptotes
// to at infinity; on PS2 content the distance term is only a few percent on top
// of it, so anything that does not subtract A first is swamped by the constant
// and comes out flat. Removing A and taking the reciprocal gives real linear
// view-space distance - which is what AO, reflections and motion validation all
// assume they are being handed.
//
// Doing it here rather than in a shader means every ReShade effect gets correct
// depth with no edits, no #defines and no dependency on any particular shader
// pack. Since Oct 23 2026 the result is also written the way ReShade's own depth
// settings will read it back (Encode, in the conversion pass), so those settings can
// stay at ReShade's defaults. With the old neutral values (multiplier 1, not reversed,
// far plane 1) the published depth is exactly the distance, as it always was.
// inspector
// Every view here is derived from depth alone - including the reconstructed
// normals - so it works in any game with any shader pack, or none at all.
static int         g_dbg_view      = 0;
static float       g_dbg_log_min   = -20.0f;
static float       g_dbg_log_max   = -8.0f;
static float       g_dbg_near_gain = 1000.0f;
static float       g_dbg_bands     = 512.0f;
static float       g_dbg_hud_cut   = 0.021f;
static float       g_dbg_fov       = 70.0f;

// automatic calibration
// Idle does nothing at all: no passes, no readbacks. Everything below runs only
// while the user has asked for it, which is also why a brief stall is acceptable
// and the asynchronous readback machinery a background version would need is
// simply absent.
enum cal_state { CAL_IDLE = 0, CAL_CAPTURING, CAL_SHOWING };
static int    g_cal_state   = CAL_IDLE;
static double g_cal_started = 0.0;
static float  g_cal_seconds = 2.0f;
static int    g_cal_frames  = 0;
// The aspect of the rectangle the emulator actually draws its picture into. Lumenite
// derives horizontal field of view from it, and letterboxing means the buffer's own
// aspect is the wrong answer - so hand it the measured one rather than making anyone
// type it. Pushed into every Lumenite effect, because each includes its own copy.
static float  g_picture_aspect  = 0.0f;
static float  g_pushed_aspect   = -1.0f;
static int    g_aspect_tries    = 0;
static int    g_cal_flip_votes  = 0;   // frames whose clear value sat at the MAXIMUM
static int    g_cal_vote_frames = 0;
static float  g_cal_result_zero = 0.0f, g_cal_result_far = 0.0f;
static bool   g_cal_ok = false;
static char   g_cal_status[256] = "";
static std::vector<float> g_cal_mins, g_cal_maxs, g_cal_huds;
static int    g_dbg_view_before = 0;

// The overlay cutoff falls out of the same measurement. Calibration already has to
// separate the HUD from real geometry to keep it out of the near end, so by the time
// the constants are solved it is holding both raw values - and once A and F exist,
// both convert straight to linear depth. That places the cutoff instead of leaving a
// slider for the user to hunt, and re-places it whenever calibration moves, which
// matters because the cutoff is in depth units and so goes stale otherwise.
static float  g_cal_hud_depth = 0.0f, g_cal_geo_depth = 0.0f;
// What the measurement actually saw, kept so a bad result can be diagnosed
// instead of guessed at. These are the numbers everything else is derived from.
static float  g_cal_raw_far = 0.0f, g_cal_raw_near = 0.0f;
static bool   g_cfg_snap_zero = true;
static int    g_cutoff_tries  = 0;      // frames left to keep trying to reach the shader

static bool        g_cfg_linearise = true;
// Diagnostic switch. The cfg file is polled for changes, so this can be flipped WHILE the
// game runs: boot with capture=0, get in-world, then set capture=1 to see whether the
// problem belongs to steady-state capture or to a load/resource-recreation transition.
static bool        g_cfg_capture = true;
// Diagnostic gates for the passive tracking path, so the add-on can be bisected from the cfg
// file without a rebuild. All three default on; set any to 0 to make those handlers return
// immediately and touch nothing. Turning them all off leaves the add-on loaded and hooked but
// completely inert, which should behave like removing the file.
static bool        g_cfg_count_draws  = true;   // draw / draw_indexed / draw_indirect
static bool        g_cfg_track_binds  = true;   // bind_viewports / bind_scissors / bind_rtv_dsv
static bool        g_cfg_track_passes = true;   // begin_render_pass / end_render_pass
static bool        g_cfg_track_cmd_lifetime = true; // init_command_list / destroy_command_list: frees each
                                                    // list's cmd_state when the list goes away
static bool        g_cfg_auto_bars    = true;   // Auto aspect finds the picture from the black bars (0 = the old viewport method)
// The remaining registrations. With all seven gates off the DLL is loaded but subscribes to
// nothing at all, which isolates "the add-on's presence" from "anything it hooks".
static bool        g_cfg_hook_present   = true;   // present
static bool        g_cfg_hook_effects   = true;   // reshade_begin/reloaded/finish_effects
static bool        g_cfg_hook_swapchain = true;   // init_swapchain + destroy_device
static bool        g_cfg_hook_overlay   = true;   // the ImGui settings panel
// The only two things on_present does that reach outside our own memory: re-reading the cfg
// file, and asking the swapchain for its HWND to pick a game profile. Gated separately so the
// function can be bisected without a rebuild.
static bool        g_cfg_match_frame    = true;   // prefer a depth target the size of the back buffer
// Grab the depth target the instant before the game clears it. The contents at that moment
// are the finished frame, which sidesteps having to guess which of several passes holds the
// complete scene. This is the same trick ReShade's own Generic Depth calls "copy depth buffer
// before clear operations", and it is what makes Fallout 3, Arkham Asylum and Mass Effect work.
static bool        g_cfg_whole_scene    = false;  // capture at clear boundaries, not by draw count
static bool        g_cfg_poll_config    = true;   // ReadConfig() every 180 frames
static bool        g_cfg_poll_title     = false;  // periodic re-check; off by default, use the button
static bool        g_title_recheck      = false;  // set by the overlay button, cleared once handled

// Saving a profile and CHOOSING it are two different things, and only the first was
// working. With title polling off nothing ever read the window title, so a measured
// game sat in the cfg unused until the Check game button was pressed. One read, once,
// a few seconds after the swapchain settles, fixes that without reintroducing the
// polling: the crash came from reading the title repeatedly DURING level loads, when
// the UI thread was not pumping messages. At startup it is idle.
static const uint64_t kTitleOnceFrame = 180;

// Effects before the HUD
// A game's HUD is drawn into the colour buffer before ReShade ever runs at present, so
// no depth trick can exclude it: at HUD pixels the depth is whatever is behind them.
// The only real fix is to run the effects EARLIER, just before the HUD pass starts.
// ReShade's own header names this as the intended use of render_effects.
//
// The HUD pass is found by watching, not by guessing: it is the frame-sized colour pass
// with NO depth-stencil attached that draws the most. In BioShock Remastered the tail of
// a frame is the world (606 draws), the weapon (91), then one colour-only pass of 311,
// then present. Which pass that is stays stable frame to frame, so the index learned
// from one frame is used on the next.
static bool        g_cfg_hud_split      = false;
static uint32_t    g_cfg_hud_min_draws  = 32;
static float       g_cfg_zero_log2 = -9.005f;   // log2 of A
static float       g_cfg_far_scale = 154000.0f; // what distance reads as 1.0
static float       g_cfg_place_scale = 1.0f;
// Some buffers count the other way: 0 at the eye and 1 at infinity, rather than the
// console convention of a large value up close. DuckStation's reconstructed PS1 depth
// is one - it writes the projected NDC z, so the sky sits at exactly 1.0. Flipping
// the raw value on the way in puts every buffer in the same orientation, and leaves
// the solve, the debug views and the shaders untouched.
static bool        g_cfg_flip_depth = false;
// The shape of the published depth, between true distance and the buffer's own curve.
//
// 1.0 is physically correct: proportional to distance. Console depth is hyperbolic
// compressed far away, stretched up close - and the effects were written against PC
// buffers that are hyperbolic too. Correct depth is not automatically the depth they
// look best on, so this is a dial rather than a fix, and it is saved per game.
static float       g_cfg_depth_curve = 1.0f;
// Some buffers are ALREADY linear in distance, and must not be put through the
// reciprocal at all.
//
// A PS2 or GameCube depth buffer is 1/z: raw = A + B/distance, with a large constant A
// that swamps the distance term. Removing A and taking the reciprocal is the whole
// point of this tool on those platforms.
//
// DuckStation's PGXP depth is not that. It writes the GTE's own Z divided by 65535
// (cpu_pgxp.cpp: *out_w = vert->z / GTE::MAX_Z), and the vertex shader passes it
// straight through as z/w - so the stored value IS normalised view distance, small
// near and large far. Running the 1/z conversion over it turns correct linear depth
// INTO a hyperbolic curve, which is worse than doing nothing.
static bool        g_cfg_linear_buffer = false;
static float       g_cfg_place_x = 0.0f, g_cfg_place_y = 0.0f;
static float       g_cfg_overlay_cutoff = -1.0f; // pushed into Lumenite; <0 = never measured
static uint64_t    g_cfg_stamp  = 0;
static char        g_cfg_path[MAX_PATH] = {};
static char        g_cfg_text[32] = "auto";

static char g_status[192] = "waiting for a depth pass...";

// Overlay combo <-> config value
static const char *kAspectLabel[] = { "Auto", "16:9", "4:3", "Full frame (stretched)",
                                      "Match depth buffer", "16:10" };
static const char *kAspectText [] = { "auto", "16:9", "4:3", "full", "buffer", "16:10" };
// -2 means "whatever shape the captured buffer is". Dolphin displays its picture at
// the raw EFB shape (640x528 = 1.212), not at the game's nominal 4:3, so no fixed
// ratio in this list can ever match it - but the buffer we captured always can.
static const float kAspectValue[] = { 0.0f, 16.0f/9.0f, 4.0f/3.0f, -1.0f, -2.0f, 16.0f/10.0f };
static int g_aspect_index = 0;

// per-game profiles
//
// The constants belong to the GAME, not to the emulator or the install. A PS2 title
// picks its own Z format and projection, so switching discs invalidates every number
// here at once - and the values are four orders of magnitude apart between titles,
// which is why the leftovers do not merely look slightly wrong, they paint the screen
// a confident solid red.
//
// So the cfg is sectioned by disc serial. The add-on applies the section matching
// whatever is running, and a title it has never measured gets NO numbers rather than
// somebody else's. Sections accumulate as games are measured, which also makes the
// file the shippable table: seed it with known-good values and most people never
// press the button at all.
struct GameProfile
{
    float zero       = -9.005f;
    float far_scale  = 154000.0f;
    float cutoff     = -1.0f;      // <0 = never measured
    char  aspect[16] = "auto";
    char  name[64]   = "";
    bool  accumulate = true;
    bool  sparse_refresh = false;   // see g_cfg_sparse_refresh
    uint32_t min_draws = 0;         // see g_cfg_min_draws
    bool  measured   = false;
    // Manual placement nudge. Some emulators - Dolphin among them - blit their final
    // picture with a full-frame viewport and scissor, so nothing in the graphics stream
    // says where the picture actually is. Auto-detection cannot win there, so this is
    // the escape hatch: scale and shift the placement by eye, once, per game.
    float place_scale = 1.0f;
    bool  flip = false;
    float curve = 1.0f;
    bool  linear_buffer = false;
    float place_x = 0.0f, place_y = 0.0f;   // fractions of the frame
};
// the shipped table
//
// Games measured by hand and known good, compiled in rather than kept in the cfg.
// The cfg belongs to the player and gets rewritten as they tune, so it cannot also
// be the place the original numbers live - one tweak and they are gone. Built in,
// they survive anything the player does and there is always a way back to them.
//
// A built-in can say what a cfg section says about the game: PS1's linear buffer and
// draw filter, a flipped buffer, sparse refresh, and whether to merge every pass.
// (That last one is a property of the GAME, not a preference. THUG2 accumulates depth
// across its GS flushes, so merging fills the holes where the player model and near
// ground are drawn late. Spider-Man 2 clears depth before its trailing passes, so the
// same merge overwrites a 963-draw scene with an 18-draw remnant. One global setting
// cannot be right for both.)
//
// The picture aspect stays on Auto. It belongs to the player's emulator settings, not
// to the game: THUG2 is 16:9 on one PCSX2 and 4:3 on a default one. The exception is a
// picture Auto has no shape for: Dolphin shows Skies of Arcadia at the raw EFB shape
// (1.212), so that one says "buffer".
//
// Settings that describe the install rather than the game (whole_scene for BioShock)
// cannot live here. The README covers those.
//
// To add a game: measure it, then copy its cfg section into a line here. Key on the
// emulator's window title (or the disc serial if it shows one); matching is
// case-insensitive. Plain positional fields on purpose: MSVC rejects designated
// initializers over GameProfile's char-array defaults.
struct BuiltinProfile
{
    const char *key;
    float zero, far_scale, cutoff;
    const char *aspect;
    bool accumulate, sparse_refresh;
    uint32_t min_draws;
    bool flip, linear_buffer;
};
static const BuiltinProfile kBuiltins[] =
{
    //  game (window title key)               zero        far_scale       cutoff    aspect    merge  sparse draws flip   linear
    // PCSX2 (Dan's cfg, 23 Oct 2026)
    { "Crazy Frog Racer 2",                   -37.6077f, 215497040.0f, 0.00285f, "auto",   false, false, 0, false, false },
    { "Final Fantasy X",                      -29.6204f,    849209.6f, 0.00000f, "auto",   false, false, 0, false, false },
    { "Front Mission 4",                      -29.3229f,    691012.0f, 0.00000f, "auto",   false, false, 0, false, false },
    { "Grand Theft Auto - San Andreas",       -29.6061f,    840846.9f, 0.00000f, "auto",   false, true,  0, false, false },
    { "Guitar Hero II",                       -22.4625f,  25001026.0f, 0.01161f, "auto",   false, false, 0, false, false },
    { "Kingdom Hearts",                       -30.9638f,   2154970.2f, 0.00000f, "auto",   false, false, 0, false, false },
    { "Metal Gear Solid 3 - Snake Eater",     -17.3314f,    715200.1f, 0.00423f, "auto",   true,  false, 0, false, false },
    { "SoulCalibur III",                      -18.2014f,    667347.2f, 0.00000f, "auto",   false, false, 0, false, false },
    { "Spider-Man 2",                         -28.4753f,    383999.8f, 0.01488f, "auto",   false, false, 0, false, false },
    { "Star Wars - Battlefront II",           -13.2314f,       546.0f, 0.00000f, "auto",   false, true,  0, false, false },
    { "Tekken Tag Tournament",                -38.0000f,  10121200.0f, 0.00000f, "auto",   true,  false, 0, false, false },
    { "TimeSplitters 2",                      -28.8678f,    504058.6f, 0.00423f, "auto",   true,  false, 0, false, false },
    { "Tony Hawk's Underground 2",             -9.0000f,   1026570.9f, 0.00499f, "auto",   true,  false, 0, false, false },
    { "WWE SmackDown! vs. RAW",               -16.7277f,    290490.6f, 0.00000f, "auto",   false, false, 0, false, false },
    // DuckStation (Dan's cfg, 23 Oct 2026)
    { "Brave Fencer Musashi",                   0.0000f,        73.7f, 0.00000f, "auto",   false, false, 1, false, true  },
    { "Crash Bandicoot",                        0.0000f,         7.9f, 0.00000f, "auto",   false, false, 1, false, true  },
    { "MediEvil",                               0.0000f,        16.8f, 0.00000f, "auto",   false, false, 1, false, true  },
    { "Silent Hill",                            0.0000f,        22.0f, 0.00000f, "auto",   false, false, 1, false, true  },
    { "Spyro the Dragon",                     -29.9316f,   1053660.6f, 0.00000f, "auto",   true,  false, 1, true,  true  },
    { "The Legend of Dragoon (Disc 1)",         0.0000f,         3.0f, 0.00000f, "auto",   true,  false, 0, false, true  },
    { "Tomb Raider II - Starring Lara Croft",   0.0000f,         5.3f, 0.00000f, "auto",   false, false, 1, false, true  },
    // Dolphin (Dan's cfg, 23 Oct 2026)
    { "Mario Kart: Double Dash!! (GM4E01)",   -22.0890f,      4590.4f, 0.00000f, "auto",   false, false, 0, false, false },
    { "Skies of Arcadia: Legends (GEAE8P)",    -9.3682f,       711.5f, 0.00000f, "buffer", false, false, 0, false, false },
    // PC (Dan's cfg, 23 Oct 2026). Also needs whole_scene=1, which is an install setting.
    { "Bioshock",                             -10.9411f,      1287.8f, 0.00625f, "auto",   false, false, 0, true,  false },
};

static const BuiltinProfile *FindBuiltin(const std::string &key)
{
    for (const BuiltinProfile &b : kBuiltins)
        if (_stricmp(b.key, key.c_str()) == 0)
            return &b;
    return nullptr;
}

enum profile_source { SRC_NONE = 0, SRC_BUILTIN, SRC_USER };
static int g_game_source = SRC_NONE;

static std::map<std::string, GameProfile> g_profiles;
static std::string g_game_key;                 // disc serial, e.g. SLUS-20965
static char        g_window_title[256] = "";
static bool        g_game_known = false;
static bool        g_cfg_loaded = false;   // the file on disk has been read at least once

static void WriteConfig();

// Emulators put the disc serial in the window title. Two spellings are in the wild
// SLUS-20965 and SLUS_209.65 - and both normalise to the first.
static bool ExtractSerial(const char *title, std::string &out)
{
    for (const char *p = title; p[0] && p[1] && p[2] && p[3] && p[4]; ++p)
    {
        if (!isalpha((unsigned char)p[0]) || !isalpha((unsigned char)p[1]) ||
            !isalpha((unsigned char)p[2]) || !isalpha((unsigned char)p[3]))
            continue;
        if (p[4] != '-' && p[4] != '_') continue;
        char digits[8] = {}; int n = 0;
        for (const char *q = p + 5; *q && n < 6; ++q)
        {
            if (*q == '.') continue;
            if (*q < '0' || *q > '9') break;
            digits[n++] = *q;
        }
        if (n != 5) continue;
        char buf[24];
        sprintf_s(buf, "%c%c%c%c-%s",
                  (char)toupper((unsigned char)p[0]), (char)toupper((unsigned char)p[1]),
                  (char)toupper((unsigned char)p[2]), (char)toupper((unsigned char)p[3]), digits);
        out = buf;
        return true;
    }
    return false;
}

// Emulators lay their title bars out differently: PCSX2 leads with the game,
// Dolphin leads with its own version ("Dolphin 2606a | ... | Game | 60 FPS"), and
// most append a live framerate. So rather than assume a position, split on " | ",
// throw away the segments that are demonstrably not a game name, and keep the
// longest of what remains - a game title is almost always the longest field, and
// it is the only one that does not change while you play.
static bool LooksLikeChrome(const std::string &t)
{
    static const char *kJunk[] = { "fps", "vps", "direct3d", "d3d1", "vulkan", "opengl",
                                   "software", "metal", "jit", "interpreter", "dsp",
                                   "speed", "paused", "hle", "lle" };
    std::string low;
    for (char c : t) low += (char)tolower((unsigned char)c);
    for (const char *j : kJunk)
        if (low.find(j) != std::string::npos) return true;
    if (t.find('%') != std::string::npos) return true;
    // "60/60", "1.00x" and similar: nothing but digits and punctuation.
    bool letter = false;
    for (char c : t) if (isalpha((unsigned char)c)) { letter = true; break; }
    return !letter;
}

static bool KeyFromTitle(const char *title, std::string &out)
{
    std::vector<std::string> parts;
    {
        const std::string t = title;
        size_t at = 0;
        for (;;)
        {
            const size_t p = t.find(" | ", at);
            parts.push_back(t.substr(at, p == std::string::npos ? std::string::npos : p - at));
            if (p == std::string::npos) break;
            at = p + 3;
        }
    }

    std::string best;
    for (const std::string &raw : parts)
    {
        // Strip anything that would break a [section] header, collapse whitespace.
        std::string k; bool space = false;
        for (char c : raw)
        {
            if (c == '[' || c == ']' || c == '=' || c == '#' || c == ';' || (unsigned char)c < 32)
                continue;
            if (c == ' ' || c == '\t') { space = !k.empty(); continue; }
            if (space) { k += ' '; space = false; }
            k += c;
        }
        if (k.size() < 3) continue;
        if (parts.size() > 1 && LooksLikeChrome(k)) continue;
        if (k.size() > best.size()) best = k;
    }

    if (best.size() < 3) return false;
    // What the title says when no disc is loaded: the emulator naming itself.
    if (_strnicmp(best.c_str(), "PCSX2", 5) == 0 || _strnicmp(best.c_str(), "Dolphin", 7) == 0)
        return false;
    out.swap(best);
    return true;
}

static void ParseAspect(const char *v, float &value, int &index, char *text, size_t text_bytes)
{
    strncpy_s(text, text_bytes, v, _TRUNCATE);
    if (_stricmp(v, "auto") == 0)      value = 0.0f;
    else if (_stricmp(v, "full") == 0) value = -1.0f;
    else if (_stricmp(v, "buffer") == 0) value = -2.0f;
    else
    {
        float a = 0.0f, b = 0.0f;
        if (sscanf_s(v, "%f:%f", &a, &b) == 2 && b > 0.0f) value = a / b;
        else if (sscanf_s(v, "%f", &a) == 1 && a > 0.0f)   value = a;
        else value = 0.0f;
    }
    index = 0;
    for (int i = 0; i < 6; ++i)
        if (fabsf(kAspectValue[i] - value) < 0.0005f) { index = i; break; }
}

static bool ParseBool(const char *v)
{
    return !(_stricmp(v, "0") == 0 || _stricmp(v, "off") == 0 || _stricmp(v, "false") == 0);
}

// Hand a profile to the live settings. Everything downstream reads these, so this is
// the single point where "which game is running" turns into "what depth looks like".
static void ApplyProfile(const GameProfile &p)
{
    g_cfg_zero_log2      = p.zero;
    g_cfg_far_scale      = p.far_scale;
    g_cfg_overlay_cutoff = p.cutoff;
    g_cfg_accumulate     = p.accumulate;
    g_cfg_sparse_refresh = p.sparse_refresh;
    g_cfg_min_draws      = p.min_draws;
    if (p.min_draws > 0)
        g_min_draws = p.min_draws;
    g_cfg_place_scale    = p.place_scale;
    g_cfg_flip_depth     = p.flip;
    g_cfg_depth_curve    = p.curve;
    g_cfg_linear_buffer  = p.linear_buffer;
    g_cfg_place_x        = p.place_x;
    g_cfg_place_y        = p.place_y;
    g_cutoff_tries       = 1200;
    ParseAspect(p.aspect, g_cfg_aspect, g_aspect_index, g_cfg_text, sizeof(g_cfg_text));
}

static void ApplyBuiltin(const BuiltinProfile &b)
{
    // Through ApplyProfile, so a built-in sets exactly what a cfg section would.
    GameProfile p;
    p.zero           = b.zero;
    p.far_scale      = b.far_scale;
    p.cutoff         = b.cutoff;
    strncpy_s(p.aspect, b.aspect, _TRUNCATE);
    p.accumulate     = b.accumulate;
    p.sparse_refresh = b.sparse_refresh;
    p.min_draws      = b.min_draws;
    p.flip           = b.flip;
    p.linear_buffer  = b.linear_buffer;
    ApplyProfile(p);
}

// A player's own section always wins - they measured it on their own machine, in
// their own scene - and the shipped table fills in for a game they have not touched.
static void ApplyBest(const std::string &key)
{
    auto it = g_profiles.find(key);
    if (it != g_profiles.end() && it->second.measured)
    {
        ApplyProfile(it->second);
        g_game_source = SRC_USER;
        g_game_known  = true;
        return;
    }
    if (const BuiltinProfile *b = FindBuiltin(key))
    {
        ApplyBuiltin(*b);
        g_game_source = SRC_BUILTIN;
        g_game_known  = true;
        return;
    }
    g_game_source = SRC_NONE;
    g_game_known  = false;
    g_cfg_sparse_refresh = false;
    g_cfg_min_draws = 0;
    g_cfg_overlay_cutoff = -1.0f;
    g_cutoff_tries = 1200;
}

// Anything the player changes for this game makes the settings theirs, which is what
// puts a section in the cfg and what makes "reset to tested defaults" meaningful.
static void MarkUserProfile()
{
    if (!g_game_key.empty()) { g_game_source = SRC_USER; g_game_known = true; }
}

static GameProfile &CurrentProfile()
{
    return g_profiles[g_game_key];
}

static void ReadConfig(bool announce)
{
    if (g_cfg_path[0] == 0)
        return;
    WIN32_FILE_ATTRIBUTE_DATA fad = {};
    if (!GetFileAttributesExA(g_cfg_path, GetFileExInfoStandard, &fad))
        return;
    const uint64_t stamp = ((uint64_t)fad.ftLastWriteTime.dwHighDateTime << 32) | fad.ftLastWriteTime.dwLowDateTime;
    if (stamp == g_cfg_stamp && !announce)
        return;
    g_cfg_stamp = stamp;

    FILE *f = nullptr;
    if (fopen_s(&f, g_cfg_path, "rb") != 0 || f == nullptr)
        return;

    std::map<std::string, GameProfile> profiles;
    GameProfile *section = nullptr;                 // null while in the global block
    bool parsed_accum = g_cfg_accumulate;
    char line[512];
    while (fgets(line, sizeof(line), f) != nullptr)
    {
        char *k = line;
        while (*k == ' ' || *k == '\t') ++k;
        if (*k == '#' || *k == ';' || *k == '\r' || *k == '\n' || *k == 0) continue;

        if (*k == '[')                              // [SERIAL]  optional friendly name
        {
            char *close = strchr(k, ']');
            if (close == nullptr) continue;
            *close = 0;
            std::string key = k + 1;
            char *nm = close + 1;
            while (*nm == ' ' || *nm == '\t') ++nm;
            char *ne = nm; while (*ne && *ne != '\r' && *ne != '\n') ++ne;
            *ne = 0;
            section = &profiles[key];
            // Seed from the shipped table when we know this game, and only fall back to
            // the global line otherwise. A cfg written before merging became per-game has
            // no accumulate line in its sections, so seeding from the global would stamp
            // whatever the last game happened to need onto every profile in the file.
            if (const BuiltinProfile *seed = FindBuiltin(key))
                section->accumulate = seed->accumulate;
            else
                section->accumulate = parsed_accum;
            strncpy_s(section->name, nm, _TRUNCATE);
            continue;
        }

        char *v = strchr(k, '=');
        if (v == nullptr) continue;
        char *ke = v; while (ke > k && (ke[-1] == ' ' || ke[-1] == '\t')) --ke;
        *ke = 0;
        ++v; while (*v == ' ' || *v == '\t') ++v;
        char *e = v; while (*e && *e != '\r' && *e != '\n' && *e != ';' && *e != '#') ++e;
        while (e > v && (e[-1] == ' ' || e[-1] == '\t')) --e;
        *e = 0;

        // Global block: settings that describe the install, not the disc.
        if (section == nullptr)
        {
            if (_stricmp(k, "accumulate") == 0)      { parsed_accum = ParseBool(v); continue; }
            if (_stricmp(k, "linearise") == 0)       { g_cfg_linearise = ParseBool(v); continue; }
            if (_stricmp(k, "capture") == 0)         { g_cfg_capture = ParseBool(v); continue; }
            if (_stricmp(k, "count_draws") == 0)     { g_cfg_count_draws = ParseBool(v); continue; }
            if (_stricmp(k, "track_binds") == 0)     { g_cfg_track_binds = ParseBool(v); continue; }
            if (_stricmp(k, "track_passes") == 0)    { g_cfg_track_passes = ParseBool(v); continue; }
            if (_stricmp(k, "track_cmd_lifetime") == 0) { g_cfg_track_cmd_lifetime = ParseBool(v); continue; }
            if (_stricmp(k, "auto_bars") == 0)       { g_cfg_auto_bars = ParseBool(v); continue; }
            if (_stricmp(k, "hook_present") == 0)    { g_cfg_hook_present = ParseBool(v); continue; }
            if (_stricmp(k, "hook_effects") == 0)    { g_cfg_hook_effects = ParseBool(v); continue; }
            if (_stricmp(k, "hook_swapchain") == 0)  { g_cfg_hook_swapchain = ParseBool(v); continue; }
            if (_stricmp(k, "hook_overlay") == 0)    { g_cfg_hook_overlay = ParseBool(v); continue; }
            if (_stricmp(k, "match_frame_size") == 0) { g_cfg_match_frame = ParseBool(v); continue; }
            if (_stricmp(k, "whole_scene") == 0)     { g_cfg_whole_scene = ParseBool(v); continue; }
            if (_stricmp(k, "effects_before_hud") == 0) { g_cfg_hud_split = ParseBool(v); continue; }
            if (_stricmp(k, "poll_config") == 0)     { g_cfg_poll_config = ParseBool(v); continue; }
            if (_stricmp(k, "poll_title") == 0)      { g_cfg_poll_title = ParseBool(v); continue; }
            if (_stricmp(k, "snap_zero_point") == 0) { g_cfg_snap_zero = ParseBool(v); continue; }
            continue;
        }

        // Section block: everything measured per game.
        if (_stricmp(k, "picture_aspect") == 0) { strncpy_s(section->aspect, v, _TRUNCATE); continue; }
        if (_stricmp(k, "accumulate") == 0)     { section->accumulate = ParseBool(v); continue; }
        if (_stricmp(k, "sparse_refresh") == 0) { section->sparse_refresh = ParseBool(v); continue; }
        if (_stricmp(k, "min_draws") == 0) { int x; if (sscanf_s(v, "%d", &x) == 1 && x >= 0 && x <= 1000) section->min_draws = uint32_t(x); continue; }
        if (_stricmp(k, "linear_buffer") == 0) { int x; if (sscanf_s(v, "%d", &x) == 1) section->linear_buffer = (x != 0); continue; }
        if (_stricmp(k, "depth_curve") == 0) { float x; if (sscanf_s(v, "%f", &x) == 1 && x > 0.05f && x < 8.0f) section->curve = x; continue; }
        if (_stricmp(k, "flip_depth") == 0) { int x; if (sscanf_s(v, "%d", &x) == 1) section->flip = (x != 0); continue; }
        if (_stricmp(k, "place_scale") == 0) { float x; if (sscanf_s(v, "%f", &x) == 1 && x > 0.1f) section->place_scale = x; continue; }
        if (_stricmp(k, "place_x") == 0)     { float x; if (sscanf_s(v, "%f", &x) == 1) section->place_x = x; continue; }
        if (_stricmp(k, "place_y") == 0)     { float x; if (sscanf_s(v, "%f", &x) == 1) section->place_y = x; continue; }
        if (_stricmp(k, "zero_point_log2") == 0)
        { float x; if (sscanf_s(v, "%f", &x) == 1) { section->zero = x; section->measured = true; } continue; }
        if (_stricmp(k, "far_scale") == 0)
        { float x; if (sscanf_s(v, "%f", &x) == 1 && x > 0.0f) section->far_scale = x; continue; }
        if (_stricmp(k, "overlay_cutoff") == 0)
        { float x; if (sscanf_s(v, "%f", &x) == 1) section->cutoff = x; continue; }
    }
    fclose(f);

    g_profiles.swap(profiles);

    if (parsed_accum != g_cfg_accumulate || announce)
    {
        g_cfg_accumulate = parsed_accum;
        Log("[depth-charge] accumulate = %s", g_cfg_accumulate ? "on (merge every pass into the frame's depth target)" : "off (busiest pass only)");
    }

    // Re-apply whatever now applies to the running game.
    if (!g_game_key.empty())
        ApplyBest(g_game_key);
    g_cfg_loaded = true;
    if (announce)
        Log("[depth-charge] config has %zu game profile(s)", g_profiles.size());
}

// Called when the window title changes, which on PCSX2 happens as a disc is loaded.
static void SelectGame(const char *title)
{
    if (title == nullptr || strcmp(title, g_window_title) == 0)
        return;
    strncpy_s(g_window_title, title, _TRUNCATE);

    // A disc serial is the sharper key when the emulator offers one, but most just
    // show the game's name, so fall back to that rather than giving up on profiles.
    std::string key;
    if (!ExtractSerial(title, key) && !KeyFromTitle(title, key))
    {
        Log("[depth-charge] window title \"%s\" is not usable as a game key - staying on %s",
            title, g_game_key.empty() ? "(nothing)" : g_game_key.c_str());
        return;
    }
    if (key == g_game_key)
        return;

    g_game_key = key;
    ApplyBest(key);
    // Refusing to inherit matters more than it sounds: the previous game's constants
    // would otherwise be applied with total confidence and be orders of magnitude out.
    Log("[depth-charge] game \"%s\" from title \"%s\" - %s", key.c_str(), title,
        g_game_source == SRC_USER    ? "your measured profile" :
        g_game_source == SRC_BUILTIN ? "tested default from the shipped table" :
                                       "not measured, waiting for Depth Finder");
    if (g_cfg_min_draws > 0)
        Log("[depth-charge] draw filter %u draws from the start (this game's min_draws)", g_min_draws);
}

static void WriteConfig()
{
    if (g_cfg_path[0] == 0) return;

    // Never overwrite a config that has not been read. Otherwise a write triggered
    // before the first read - or while the file is unreadable - would silently
    // replace a hand-seeded table of game profiles with an empty one.
    WIN32_FILE_ATTRIBUTE_DATA exists = {};
    if (!g_cfg_loaded && GetFileAttributesExA(g_cfg_path, GetFileExInfoStandard, &exists))
        return;

    // Fold the live settings back into the running game's section before writing, so
    // the file always reflects what is on screen.
    if (!g_game_key.empty() && g_game_source == SRC_USER)
    {
        GameProfile &p = CurrentProfile();
        p.zero      = g_cfg_zero_log2;
        p.far_scale = g_cfg_far_scale;
        p.cutoff    = g_cfg_overlay_cutoff;
        p.accumulate = g_cfg_accumulate;
        p.sparse_refresh = g_cfg_sparse_refresh;
        p.min_draws = g_cfg_min_draws;
        p.place_scale = g_cfg_place_scale;
        p.flip        = g_cfg_flip_depth;
        p.curve       = g_cfg_depth_curve;
        p.linear_buffer = g_cfg_linear_buffer;
        p.place_x = g_cfg_place_x;
        p.place_y = g_cfg_place_y;
        p.measured  = true;
        strncpy_s(p.aspect, kAspectText[g_aspect_index], _TRUNCATE);
    }

    FILE *f = nullptr;
    if (fopen_s(&f, g_cfg_path, "wb") != 0 || f == nullptr) return;
    fprintf(f,
        "# Greenskull's Depth Charge\r\n"
        "#\r\n"
        "# Settings above the first [SERIAL] line describe this install. Everything\r\n"
        "# under a [SERIAL] line was measured for that disc and applies only to it,\r\n"
        "# a PS2 game picks its own Z format, so the numbers differ by orders of\r\n"
        "# magnitude between titles and are never interchangeable.\r\n"
        "#\r\n"
        "# The add-on matches the disc serial in the emulator's window title, applies\r\n"
        "# that section, and adds a new one the first time you run Depth Finder on a\r\n"
        "# game it has not seen. Sections can also be pasted in from someone else,\r\n"
        "# a measured game never needs measuring again.\r\n"
        "#\r\n"
        "# accumulate =\r\n"
        "#   1  keep re-copying the frame's depth target as later passes add to it.\r\n"
        "#      PCSX2 splits a frame across many passes; the player model and the\r\n"
        "#      nearest ground are drawn after the bulk of the world, so taking only\r\n"
        "#      the busiest pass leaves holes exactly where those late draws are.\r\n"
        "#   0  busiest pass of the frame only.\r\n"
        "#\r\n"
        "# linearise =\r\n"
        "#   1  convert captured depth to true linear distance before publishing it,\r\n"
        "#      so every ReShade effect gets correct depth with no shader edits. It is\r\n"
        "#      written so that ReShade's own depth settings read it back correctly,\r\n"
        "#      whatever they are.\r\n"
        "#   0  publish the raw depth exactly as captured.\r\n"
        "#\r\n"
        "# whole_scene =\r\n"
        "#   1  take depth from the last scene to finish before the frame is shown,\r\n"
        "#      instead of from the busiest single pass. A scene is everything drawn\r\n"
        "#      between two clears of the depth target. Games that draw the\r\n"
        "#      first-person weapon into the same buffer after clearing it need this:\r\n"
        "#      the weapon is a fixed number of draws, so whenever the view is sparse\r\n"
        "#      (facing a wall, looking at the floor) it outweighs the level and the\r\n"
        "#      effects get the gun. Only for games that clear depth between scenes.\r\n"
        "#   0  busiest single pass of the frame.\r\n"
        "#\r\n"
        "# snap_zero_point =\r\n"
        "#   1  when the solve lands within 2%% of a power of two, take the power of\r\n"
        "#      two. Console Z scales are built from integer ranges, so it usually is\r\n"
        "#      one, and fixing it exactly makes far_scale reproducible.\r\n"
        "#\r\n"
        "# Per game:\r\n"
        "#   picture_aspect   auto | full | 4:3 | 16:9 | any number\r\n"
        "#   accumulate       1 if depth accumulates across the game's render passes,\r\n"
        "#                    0 if it clears between them (merging then overwrites the\r\n"
        "#                    real scene with whatever the last few draws left)\r\n"
        "#   zero_point_log2  log2 of the value the buffer asymptotes to at infinity\r\n"
        "#   far_scale        which distance reads as 1.0\r\n"
        "#   overlay_cutoff   depth below which a pixel is a 2D overlay, not geometry\r\n"
        "#                    (for a Lumenite build with OVERLAY_NEAR_CUTOFF; -1 = never measured)\r\n"
        "#   sparse_refresh   1 = keep refreshing the established depth target even when a\r\n"
        "#                    view has fewer draws than the filter (looking at the sky);\r\n"
        "#                    without it the depth freezes on the last busy frame\r\n"
        "#   min_draws        the draw filter for this game from the first frame (PS1 in\r\n"
        "#                    DuckStation: 1); absent = 20, lowered automatically\r\n"
        "\r\n"
        "accumulate=%d\r\n"
        "linearise=%d\r\n"
        "snap_zero_point=%d\r\n"
        "whole_scene=%d\r\n"
        "effects_before_hud=%d\r\n",
        g_cfg_accumulate ? 1 : 0, g_cfg_linearise ? 1 : 0, g_cfg_snap_zero ? 1 : 0,
        g_cfg_whole_scene ? 1 : 0, g_cfg_hud_split ? 1 : 0);

    for (const auto &kv : g_profiles)
    {
        const GameProfile &p = kv.second;
        fprintf(f, "\r\n[%s]%s%s\r\n", kv.first.c_str(), p.name[0] ? "  " : "", p.name);
        fprintf(f, "picture_aspect=%s\r\n", p.aspect);
        fprintf(f, "zero_point_log2=%.4f\r\n", p.zero);
        fprintf(f, "far_scale=%.1f\r\n", p.far_scale);
        fprintf(f, "overlay_cutoff=%.5f\r\n", p.cutoff);
        fprintf(f, "accumulate=%d\r\n", p.accumulate ? 1 : 0);
        if (p.sparse_refresh) fprintf(f, "sparse_refresh=1\r\n");
        if (p.min_draws) fprintf(f, "min_draws=%u\r\n", p.min_draws);
        if (p.flip) fprintf(f, "flip_depth=1\r\n");
        if (p.curve != 1.0f) fprintf(f, "depth_curve=%.4f\r\n", p.curve);
        if (p.linear_buffer) fprintf(f, "linear_buffer=1\r\n");
        if (p.place_scale != 1.0f || p.place_x != 0.0f || p.place_y != 0.0f)
        {
            fprintf(f, "place_scale=%.4f\r\n", p.place_scale);
            fprintf(f, "place_x=%.4f\r\n", p.place_x);
            fprintf(f, "place_y=%.4f\r\n", p.place_y);
        }
    }
    fclose(f);
    WIN32_FILE_ATTRIBUTE_DATA fad = {};
    if (GetFileAttributesExA(g_cfg_path, GetFileExInfoStandard, &fad))
        g_cfg_stamp = ((uint64_t)fad.ftLastWriteTime.dwHighDateTime << 32) | fad.ftLastWriteTime.dwLowDateTime;
}

struct __declspec(uuid("e08dfe54-276e-4368-9156-7cfa6f42def0")) cmd_state
{
    resource_view dsv = { 0 };
    uint32_t draws = 0;
    viewport vp = {};
    bool     vp_set = false;
    // The LARGEST viewport used while this pass was open. A scene pass sets the
    // viewport many times - the world, then a mini-map, an icon, a lens flare
    // and which one happens to be last varies with draw order. Taking the last one
    // is what made the crop jump around; the largest is the picture.
    float    vp_max_w = 0.0f, vp_max_h = 0.0f;
    // True when the "depth" we are capturing is really a colour render target. The
    // PlayStation had no Z-buffer, so DuckStation reconstructs one from PGXP and
    // writes it to an R32_FLOAT colour attachment, doing its depth test in the pixel
    // shader. It is genuine per-pixel depth - just not where a GPU keeps depth.
    bool     dsv_is_color = false;
    // The largest SCISSOR used while the pass was open. Some emulators set the
    // viewport to the whole target and confine drawing with the scissor instead
    // DuckStation renders into PS1 VRAM that way, so the viewport says 1024x512 while
    // the picture is a small rectangle somewhere inside it.
    rect     sc = {};
    bool     sc_set = false;
    int32_t  sc_max_x = 0, sc_max_y = 0;
    uint32_t sc_max_w = 0, sc_max_h = 0;
};

struct device_state
{
    // The swapchain's top-level window, resolved ONCE when the swapchain is created rather
    // than asked for on every present. See on_present for why that matters.
    HWND root_hwnd = nullptr;

    // The target that won the LAST frame, and how busy it was. best_draws resets every
    // frame, so without this a game with two differently-sized depth buffers captures BOTH
    // every frame - each one beats the running best in turn. BioShock Remastered did exactly
    // that: 68k captures at 1920x1080 and 68k at 3840x2160, one pair per frame, rebuilding
    // the capture texture on every size change until resource creation started failing and a
    // stale image stayed published. Sticking to last frame's winner unless something is
    // genuinely busier keeps one target, one size, no rebuild.
    uint64_t win_target = 0;
    uint32_t win_draws  = 0;
    bool     win_fits   = false;   // did the incumbent match the back buffer's size?

    // One "scene" is everything drawn into a depth target between two clears of it.
    // A game can render several scenes per frame into the same buffer: BioShock
    // Remastered draws the world, clears, then draws the first-person weapon on its
    // own so it can never clip into the level. Both are depth; only the first is the
    // depth the effects want. Draw counts cannot tell them apart, because the weapon
    // is a constant 89 draws while the world varies with the view, so looking at a
    // wall or the floor makes the weapon the busiest pass in the frame. The clear
    // boundary can: take the biggest scene, not the biggest pass.
    // The runtime, so the HUD split can ask it to render effects mid-frame. Set on the
    // first effects event; there is no other way to reach it from a bind callback.
    effect_runtime *runtime = nullptr;

    int      co_index   = 0;    // which frame-sized colour-only pass we are on, this frame
    int      co_seen    = 0;    // how many there were last frame, for the overlay
    uint32_t hud_seen_best = 0; // and the busiest of them, qualifying or not
    int      hud_index  = -1;   // the one that drew the most, learned from the last frame
    int      hud_next   = -1;   // ...being learned for the next frame
    uint32_t hud_best   = 0;
    bool     hud_done   = false;// effects already rendered early this frame
    bool     hud_inside = false;// guard against re-entering render_effects
    uint64_t hud_frames = 0;    // how many frames actually split, for the overlay

    uint64_t iv_target = 0;     // the target whose scene is currently open
    uint32_t iv_draws  = 0;     // draws into it since its last clear
    uint32_t iv_best   = 0;     // biggest scene captured so far this frame

    // Stage 1: the depth-stencil exactly as the scene pass left it.
    resource raw = { 0 };
    uint32_t raw_w = 0, raw_h = 0;
    format   raw_fmt = format::unknown;

    // Stage 2: a screen-sized texture with that depth blitted into the rectangle
    // the emulator actually draws its picture into.
    resource      disp = { 0 };
    resource_view disp_rtv = { 0 };
    // The rectangle the emulator actually renders into, inside the capture. A
    // GameCube EFB is 640x528 but games draw 640x480 into it, so the bottom ninth is
    // never written - stretching the whole texture over the picture makes everything
    // 9% too tall. Zero means "not known, use the whole thing".
    uint32_t      raw_vp_w = 0, raw_vp_h = 0;
    uint32_t      raw_vp_x = 0, raw_vp_y = 0;
    // The largest viewport seen so far THIS frame, and the candidate we are waiting
    // to become stable. A game with several depth passes (split screen, mirrors,
    // shadow maps) offers a different viewport every pass; taking whichever came
    // last makes the crop flicker between the scene and a corner of it.
    uint32_t      vp_frame_w = 0, vp_frame_h = 0;
    uint32_t      vp_frame_x = 0, vp_frame_y = 0;
    // ...and the largest seen across a whole window of frames. Half-second stretches
    // where only a small auxiliary pass is captured are common, so a per-frame or
    // few-frame rule is not enough: the crop is the biggest viewport of the window.
    uint32_t      vp_win_w = 0, vp_win_h = 0;
    uint32_t      vp_win_x = 0, vp_win_y = 0;
    uint32_t      vp_win_frames = 0;
    resource_view disp_srv = { 0 };
    uint32_t      disp_w = 0, disp_h = 0;
    // The canvas must be rebuilt when the capture's FORMAT changes, not only its size.
    // An emulator can switch depth format at a constant resolution - DuckStation goes
    // from an R32_FLOAT mask target to a real D16 buffer, both 1024x512 - and a canvas
    // still typed for the old one reads back as zeros.
    format        disp_fmt = format::unknown;

    std::unordered_set<uint64_t> back_buffers;
    uint32_t bb_w = 0, bb_h = 0;

    subresource_box display_rect = {};
    bool display_rect_set = false;

    // Stage 3: the linearised copy actually published to ReShade.
    resource      lin = { 0 };
    resource_view lin_srv = { 0 }, lin_rtv = { 0 };
    uint32_t      lin_w = 0, lin_h = 0;

    // 16x16 tile reduction of the raw depth, plus a staging copy to read it back.
    resource      cal_rt = { 0 }, cal_stage = { 0 };
    resource_view cal_rtv = { 0 };
    pipeline        cal_pso = { 0 };
    pipeline_layout cal_layout = { 0 };
    bool            cal_tried = false;

    // Auto aspect (Oct 21 2026): a copy of the frame, a 1024x2 reduction of it, and its readback.
    resource      bar_copy = { 0 }, bar_rt = { 0 }, bar_stage = { 0 };
    resource_view bar_copy_srv = { 0 }, bar_rtv = { 0 };
    uint32_t      bar_copy_w = 0, bar_copy_h = 0;
    format        bar_copy_fmt = format::unknown;
    pipeline        bar_pso = { 0 };
    pipeline_layout bar_layout = { 0 };
    bool            bar_tried = false;
    uint32_t        bar_job_w = 0, bar_job_h = 0;   // the frame size the pending scan was taken at
    bool            bar_job_pending = false;

    pipeline        dbg_pso = { 0 };
    pipeline_layout dbg_layout = { 0 };
    format          dbg_rt_format = format::unknown;
    bool            dbg_tried = false;

    pipeline        convert_pso = { 0 };
    pipeline_layout convert_layout = { 0 };
    sampler         convert_sampler = { 0 };
    bool            convert_ready = false, convert_tried = false, convert_warned = false;

    uint32_t best_draws = 0;
    bool     best_ready = false;   // the chosen candidate is a ready-to-use picture
    uint32_t ready_grace = 0;      // frames left during which raw VRAM is ignored
    uint64_t frame_target = 0;  // the depth resource the frame's busiest pass drew into
    uint32_t merged = 0;        // passes folded into this frame's capture
    uint32_t last_merged = 0;
    uint64_t frames = 0, snapshots = 0;
    uint64_t sparse_snaps = 0;  // snapshots taken under the draw filter by sparse_refresh
    uint32_t last_draws = 0;
    bool     have_raw = false, can_blit = true, warned_blit = false;
    int      logged_rect = -1;
    uint32_t bb_binds = 0;      // how many times a back buffer was bound as a render target
    uint64_t rt_binds = 0;      // every render-target bind seen, ever
    uint64_t dsv_binds = 0;     // ...of which had a depth-stencil attached
    uint64_t rp_begins = 0;     // render passes begun (the D3D12 path)
    uint64_t rp_depth  = 0;     // ...of which declared a depth attachment
    uint64_t draw_calls = 0;    // every draw seen, ever
    uint32_t vp_logs = 0;       // diagnostic: first few blit viewports seen
    rect     last_scissor = {};
    bool     scissor_set = false;
};

// A colour target that is really a depth buffer: one channel, and either float or
// 16-bit normalised. Nothing else in an emulator's frame looks like this.
static bool LooksLikeDepthColour(format f)
{
    return f == format::r32_float || f == format::r32_typeless ||
           f == format::r16_float || f == format::r16_typeless ||
           f == format::r16_unorm;
}

// Which typed format actually reads a captured depth texture correctly.
//
// ReShade's format_to_default_typed picks the FLOAT variant of a typeless format,
// which is right for 32-bit depth (D32_FLOAT and R32_FLOAT share a bit layout) and
// wrong for 16-bit: a D16_UNORM buffer read as r16_float comes back as denormals
// near zero, which looks exactly like "nothing was drawn". Console emulators reach
// for D16 far more often than PC games do, so this has to be explicit.
static format DepthReadFormat(format typeless)
{
    switch (typeless)
    {
    case format::r16_typeless:   return format::r16_unorm;          // D16_UNORM
    case format::r24_g8_typeless:return format::r24_unorm_x8_uint;  // D24_UNORM_S8
    case format::r32_g8_typeless:return format::r32_float_x8_uint;  // D32_FLOAT_S8
    case format::r32_typeless:   return format::r32_float;          // D32_FLOAT / R32_FLOAT
    default: break;
    }
    return format_to_default_typed(typeless, 0);
}

static std::mutex g_mutex;
static std::unordered_map<device *, device_state>    g_devices;
// Each command list carries its own cmd_state as ReShade private data (Oct 20 2026). This used to
// be one global unordered_map shared by every thread, with no lock. A game that records on several
// threads at once (D3D11 deferred contexts, most D3D12 games) could make one thread grow the map
// while another was halfway through a lookup in it: a crash, or the wrong record and so the wrong
// depth, depending on timing. A command list is only recorded by one thread at a time, so state kept
// on the list itself is never shared and needs no lock. Emulators record on one thread, so nothing
// changes for them.
// Created here on first use, which also covers lists made before Depth Charge was listening.
// Freed in on_destroy_command_list (see track_cmd_lifetime).
static cmd_state &CmdState(command_list *cmd_list)
{
    if (cmd_state *const cs = cmd_list->get_private_data<cmd_state>())
        return *cs;
    return *cmd_list->create_private_data<cmd_state>();
}

// stage 1: grab the depth while it still has contents

static void snapshot(command_list *cmd_list, resource_view dsv, uint32_t draws, bool from_color,
                     bool force = false)
{
    if (!g_cfg_capture)          // diagnostic: capture disabled, touch nothing
        return;
    if (dsv == 0 || (draws == 0 && !force))
        return;

    device *const dev = cmd_list->get_device();
    std::lock_guard<std::mutex> lock(g_mutex);
    device_state &ds = g_devices[dev];

    const resource src = dev->get_resource_from_view(dsv);
    if (src == 0)
        return;

    if (g_survey_frames > 0)
    {
        const resource_desc d = dev->get_resource_desc(src);
        Log("[depth-charge]   candidate %ux%u  fmt %u  samples %u  draws %u  target %llu%s",
            d.texture.width, d.texture.height, (unsigned)d.texture.format,
            (unsigned)d.texture.samples, draws, (unsigned long long)src.handle,
            (draws >= g_min_draws) ? "   <- clears the draw filter" : "");
    }

    // Two ways a pass gets captured.
    //
    //  busier   - more draws than anything seen this frame. This is how we find
    //              the frame's real scene depth target and ignore UI, shadow and
    //              blit scratch passes.
    //
    //  refresh  - a LATER pass drawing into that same target. PCSX2 flushes the
    //              GS batch many times per frame, and depth accumulates across
    //              those flushes, so a later copy is strictly more complete than
    //              an earlier one. Keeping only the busiest pass discards
    //              everything drawn after it - which is exactly where the holes
    //              were: the player model and the nearest ground, both drawn late.
    // A third consideration, for reconstructed depth published as a colour target.
    //
    // DuckStation writes depth into emulated VRAM - a 1024x512 sheet holding two
    // framebuffers, texture pages and palettes - and then, once per frame at display
    // time, extracts the visible part of it into a display-sized texture for its own
    // post-processing. Both are R32F colour targets, so both look like depth to us,
    // and the VRAM one wins on draw count every time. The extracted one is the better
    // source by a distance: already cropped to the picture, already the right size,
    // and complete. Prefer it whenever one exists, regardless of draws.
    //
    // Scoped to colour-target depth on purpose. A real depth-stencil can legitimately
    // be larger than the window - a GameCube EFB is 640x528 for a 640x480 picture
    // so this must never affect the depth-stencil path.
    const resource_desc sd = dev->get_resource_desc(src);
    const bool ready = from_color && ds.bb_w != 0 && ds.bb_h != 0 &&
                       sd.texture.width <= ds.bb_w && sd.texture.height <= ds.bb_h;

    // Once a ready picture has been seen, stop capturing the raw sheet entirely.
    // Both appear every frame - the sheet first, with hundreds of draws, then the
    // extract - so merely preferring the extract still rebuilt the capture texture
    // twice a frame at two different sizes, which is what made the view flicker.
    if (from_color && !ready && ds.ready_grace > 0)
        return;

    const bool promotes = (ready && !ds.best_ready);          // a picture beats a sheet of VRAM
    const bool demotes  = (!ready && ds.best_ready);          // ...and never the other way round
    // A ready picture bypasses the draw filter entirely. That filter exists to tell a
    // scene pass apart from UI and scratch passes by weight of draws - but an already
    // extracted, display-sized depth image is ONE fullscreen blit, and would never
    // clear a minimum written for scene passes. Judging it by draw count is judging
    // the wrong thing: it is not competing to be the busiest pass, it is the finished
    // article.
    // Same target as last frame -> always accept, it is simply this frame's refresh.
    // A different target has to beat the incumbent's high-water mark, not just whatever has
    // been seen so far this frame.
    // A game's real scene depth is the size of the frame. A half-resolution buffer holding the
    // whole scene in one fat pass is a reflection or refraction pass - BioShock Remastered
    // renders one at 1920x1080 with 286 draws while the actual scene depth is spread over six
    // passes of 98/61/43/2/65/89 at 3840x2160, so "busiest single pass" picks the reflection
    // and the image arrives upside down. Preferring a frame-sized target settles it.
    //
    // Only a preference, never a requirement: emulator depth is routinely NOT the size of the
    // window (internal resolution, letterboxing), and when nothing matches this is inert.
    const bool fits = g_cfg_match_frame && ds.bb_w != 0 && ds.bb_h != 0 &&
                      sd.texture.width == ds.bb_w && sd.texture.height == ds.bb_h;

    const bool incumbent = (ds.win_target != 0 && src.handle == ds.win_target);
    // A frame-sized candidate outranks a non-frame-sized incumbent whatever the draw counts
    // say; the reverse can never happen. Otherwise it comes down to draws as before.
    const bool outranks  = (ds.win_target == 0) || (fits && !ds.win_fits);
    const bool outranked = (!fits && ds.win_fits);
    const bool usurps    = !outranked && (outranks || draws > ds.win_draws);
    // A capture taken just before a clear is the finished frame, so it does not have to win
    // anything on draw count. It still has to be the frame-sized target, so a reflection or
    // shadow buffer being cleared can never hijack the picture.
    // Whole-scene capture. Every pass adds to the open scene's tally; a clear ends it,
    // and ending a scene is the only thing that copies anything.
    //
    // The scene that reaches the screen is the LAST one to end before the frame is
    // presented - position in the frame, not weight of draws. Draw count was tried and
    // fails for the same reason at scene level as at pass level: run face-first into a
    // wall in BioShock Remastered and the whole level collapses to 47 draws while the
    // first-person weapon stays at its constant 89, so "biggest scene" hands the effects
    // the gun exactly when the view is simplest. Order does not care what is on screen.
    //
    // The weapon is drawn after the last clear, so its scene is still open at present
    // and is never copied. It ends at the NEXT frame's first clear, which is after the
    // effects have already run against the level.
    bool scene_take = false;
    if (g_cfg_whole_scene)
    {
        if (force)                                   // a clear: the open scene just ended
        {
            if (src.handle == ds.iv_target)
            {
                scene_take = (ds.iv_draws >= g_min_draws);
                if (scene_take) ds.iv_best = ds.iv_draws;
                if (g_survey_frames > 0)
                    Log("[depth-charge]   scene of %u draws ended%s",
                        ds.iv_draws, scene_take ? "   <- captured" : "");
                ds.iv_draws = 0;
            }
            else if (fits)                           // adopt the frame-sized target
            {
                ds.iv_target = src.handle;
                ds.iv_draws  = 0;
            }
        }
        else
        {
            if (ds.iv_target == 0 && fits) ds.iv_target = src.handle;
            if (src.handle == ds.iv_target) ds.iv_draws += draws;
        }
    }

    // sparse_refresh: the established target's own passes count whatever their size.
    const bool sparse_ok = g_cfg_sparse_refresh && incumbent && draws >= 1;
    const bool busier   = g_cfg_whole_scene
                        ? scene_take
                        : (force ? (fits || ds.win_target == src.handle)
                                 : (!demotes && (promotes ||
                                    ((draws >= g_min_draws || sparse_ok) && draws > ds.best_draws &&
                                     (incumbent || usurps)))));
    const bool refresh  = (!g_cfg_whole_scene && g_cfg_accumulate && ds.frame_target != 0 && src.handle == ds.frame_target
                          && ds.merged < 256);   // runaway guard, never reached in practice
    if (!busier && !refresh)
        return;
    if (sd.type != resource_type::texture_2d || sd.texture.samples > 1)
        return;                      // MSAA would need a resolve, not a copy

    if (ds.raw == 0 || ds.raw_w != sd.texture.width || ds.raw_h != sd.texture.height || ds.raw_fmt != sd.texture.format)
    {
        if (ds.raw != 0)
        {
            // (Oct 21 2026) The published depth can BE the capture (ds.disp == ds.raw): the full-frame
            // passthrough, and since Oct 19 the placement in the linearisation pass. Destroying the capture
            // without letting go of that alias left ds.disp naming a texture that no longer existed, and the
            // next frame destroyed it a second time, because "ds.disp != ds.raw" reads as "a canvas we own".
            // Any change in the game's depth buffer size did it: Kingdom Hearts entering a new area, Guitar
            // Hero 2 starting a song. Four crash dumps, all at those two destroy calls. ds.disp_srv is left
            // alone: it is released by the next placement, which sees the empty slot and builds a new one.
            if (ds.disp == ds.raw)
                ds.disp = { 0 };
            dev->destroy_resource(ds.raw); ds.raw = { 0 };
        }

        resource_desc bd(sd.texture.width, sd.texture.height, 1, 1,
                         format_to_typeless(sd.texture.format), 1, memory_heap::default_,
                         resource_usage::copy_dest | resource_usage::copy_source | resource_usage::shader_resource);
        if (!dev->create_resource(bd, nullptr, resource_usage::copy_source, &ds.raw))
        {
            Log("[depth-charge] could not create the %ux%u capture texture", sd.texture.width, sd.texture.height);
            return;
        }
        ds.raw_w = sd.texture.width; ds.raw_h = sd.texture.height; ds.raw_fmt = sd.texture.format;
        // A new capture size invalidates any crop measured against the old one.
        ds.raw_vp_w = 0; ds.raw_vp_h = 0; ds.raw_vp_x = 0; ds.raw_vp_y = 0;
        ds.vp_win_w = 0; ds.vp_win_h = 0; ds.vp_win_x = 0; ds.vp_win_y = 0; ds.vp_win_frames = 0;
        Log("[depth-charge] capturing %ux%u depth (format %u)%s", ds.raw_w, ds.raw_h,
            (unsigned)sd.texture.format,
            from_color ? " - from a COLOUR target (reconstructed depth, e.g. PGXP)" : "");
    }

    const resource       res [2] = { src, ds.raw };
    const resource_usage from[2] = { from_color ? resource_usage::render_target
                                                : resource_usage::depth_stencil_write,
                                     resource_usage::copy_source };
    const resource_usage to  [2] = { resource_usage::copy_source,         resource_usage::copy_dest  };
    cmd_list->barrier(2, res, from, to);
    cmd_list->copy_resource(src, ds.raw);
    cmd_list->barrier(2, res, to, from);

    // Remember the viewport this pass drew with - that, not the texture size, is the
    // part of the buffer that holds the picture.
    {
        const cmd_state &cs = CmdState(cmd_list);
        if (cs.vp_set && cs.vp_max_w >= 16.0f && cs.vp_max_h >= 16.0f)
        {
            const uint32_t vw = (uint32_t)(cs.vp_max_w + 0.5f);
            const uint32_t vh = (uint32_t)(cs.vp_max_h + 0.5f);
            if (vw <= ds.raw_w && vh <= ds.raw_h)
            {
                // Only the biggest viewport of the frame describes the picture. The
                // smaller ones are shadow, mirror and reflection passes rendering into
                // a corner of the same buffer, and cropping to one of those is what
                // produced the "zoomed into the top left" flicker.
                // The picture is the viewport clipped by the scissor. Where the two
                // agree this changes nothing; where an emulator draws into a corner of
                // a much larger target, only the scissor knows where.
                uint32_t cx = 0, cy = 0, cw2 = vw, ch2 = vh;
                if (cs.sc_max_w >= 16 && cs.sc_max_h >= 16)
                {
                    const uint32_t sx = (uint32_t)(cs.sc_max_x < 0 ? 0 : cs.sc_max_x);
                    const uint32_t sy = (uint32_t)(cs.sc_max_y < 0 ? 0 : cs.sc_max_y);
                    if (sx < vw && sy < vh)
                    {
                        const uint32_t sw2 = (cs.sc_max_w < vw - sx) ? cs.sc_max_w : (vw - sx);
                        const uint32_t sh2 = (cs.sc_max_h < vh - sy) ? cs.sc_max_h : (vh - sy);
                        if ((uint64_t)sw2 * sh2 * 4 < (uint64_t)vw * vh)   // meaningfully smaller
                        { cx = sx; cy = sy; cw2 = sw2; ch2 = sh2; }
                    }
                }
                if ((uint64_t)cw2 * ch2 > (uint64_t)ds.vp_frame_w * ds.vp_frame_h)
                {
                    ds.vp_frame_w = cw2; ds.vp_frame_h = ch2;
                    ds.vp_frame_x = cx;  ds.vp_frame_y = cy;
                }
                if (g_survey_frames > 0)
                    Log("[depth-charge] render viewport %ux%u inside a %ux%u capture (%u draws)",
                        vw, vh, ds.raw_w, ds.raw_h, draws);
            }
        }
    }

    if (busier)
    {
        ds.best_draws = g_cfg_whole_scene ? ds.iv_best : draws;
        ds.best_ready = ready; ds.frame_target = src.handle;
        if (ds.win_target != src.handle) { ds.win_draws = 0; ds.win_fits = fits; }
        ds.win_target = src.handle;
        ds.win_fits   = fits;
        if (draws > ds.win_draws) ds.win_draws = draws;
    }
    if (ready)  ds.ready_grace = 300;
    ds.merged++;
    ds.last_draws = draws;
    ds.have_raw = true;
    ds.snapshots++;
    if (busier && draws < g_min_draws) ds.sparse_snaps++;
}

// automatic calibration
//
// What the two sliders need is the smallest and largest raw depth in the scene:
// the asymptote comes from the farthest surface, the scale from the nearest. Both
// are frame-wide extremes, which a pixel shader cannot report - so reduce to a
// 16x16 grid of per-tile min/max and read those 256 pairs back.
//
// Outliers are rejected twice. Within a frame, by taking a percentile across the
// 256 tiles rather than the absolute extreme, so a lens flare or a stray sprite
// cannot set the answer. Across frames, by taking a percentile again, so a single
// bad frame - a cutscene, a loading screen - cannot either. That matters because
// the extremes are one-way: without it, one bad frame produces a confident wrong
// answer that never recovers.

static const int  kCalTiles = 16;
static const char *kCalHLSL = R"__(
Texture2D<float>  src : register(t0);
SamplerState      smp : register(s0);

void vs_main(uint id : SV_VertexID, out float4 pos : SV_Position, out float2 uv : TEXCOORD)
{
    uv  = float2((id == 2) ? 2.0 : 0.0, (id == 1) ? 2.0 : 0.0);
    pos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
}

float2 ps_main(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    // Each output texel scans a grid of samples across its own slice of the screen.
    const int N = 12;
    float2 base = floor(pos.xy) / 16.0;
    float mn = 1e9, mx = 0.0;
    [loop] for (int y = 0; y < N; y++)
    [loop] for (int x = 0; x < N; x++)
    {
        float2 t = base + float2((x + 0.5) / (N * 16.0), (y + 0.5) / (N * 16.0));
        float d = src.SampleLevel(smp, t, 0);
        if (d <= 0.0) continue;              // nothing drawn: sky, or an overlay hole
        mn = min(mn, d);
        mx = max(mx, d);
    }
    if (mx <= 0.0) return float2(0.0, 0.0);  // tile had no geometry at all
    return float2(mn, mx);
}
)__";

static bool BuildCalibrator(device *dev, device_state &ds)
{
    if (ds.cal_pso != 0) return true;
    if (ds.cal_tried)    return false;
    ds.cal_tried = true;

    const device_api api = dev->get_api();
    if (api != device_api::d3d10 && api != device_api::d3d11 && api != device_api::d3d12)
    {
        Log("[depth-charge] automatic calibration needs Direct3D 10/11/12");
        return false;
    }

    ID3DBlob *vs = nullptr, *ps = nullptr, *err = nullptr;
    if (FAILED(D3DCompile(kCalHLSL, strlen(kCalHLSL), nullptr, nullptr, nullptr,
                          "vs_main", "vs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &vs, &err)) ||
        FAILED(D3DCompile(kCalHLSL, strlen(kCalHLSL), nullptr, nullptr, nullptr,
                          "ps_main", "ps_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &ps, &err)))
    {
        Log("[depth-charge] could not compile the calibration shader: %s",
            err ? (const char *)err->GetBufferPointer() : "(no message)");
        if (vs) vs->Release(); if (ps) ps->Release(); if (err) err->Release();
        return false;
    }
    if (err) { err->Release(); err = nullptr; }

    descriptor_range srv = {};
    srv.binding = 0; srv.dx_register_index = 0; srv.count = 1;
    srv.type = descriptor_type::shader_resource_view; srv.visibility = shader_stage::pixel;
    descriptor_range smp = {};
    smp.binding = 0; smp.dx_register_index = 0; smp.count = 1;
    smp.type = descriptor_type::sampler; smp.visibility = shader_stage::pixel;
    const pipeline_layout_param params[2] = { pipeline_layout_param(srv), pipeline_layout_param(smp) };
    if (!dev->create_pipeline_layout(2, params, &ds.cal_layout))
    { vs->Release(); ps->Release(); return false; }

    shader_desc vsd = {}; vsd.code = vs->GetBufferPointer(); vsd.code_size = vs->GetBufferSize();
    shader_desc psd = {}; psd.code = ps->GetBufferPointer(); psd.code_size = ps->GetBufferSize();
    const format rtf = format::r32g32_float;
    blend_desc bl = {}; rasterizer_desc rs = {}; rs.cull_mode = cull_mode::none;
    depth_stencil_desc dss = {}; dss.depth_enable = false; dss.stencil_enable = false;
    const primitive_topology topo = primitive_topology::triangle_list;
    const uint32_t mask = 0xFFFFFFFF;
    const pipeline_subobject so[] = {
        { pipeline_subobject_type::vertex_shader,         1, (void *)&vsd },
        { pipeline_subobject_type::pixel_shader,          1, (void *)&psd },
        { pipeline_subobject_type::render_target_formats, 1, (void *)&rtf },
        { pipeline_subobject_type::blend_state,           1, (void *)&bl  },
        { pipeline_subobject_type::rasterizer_state,      1, (void *)&rs  },
        { pipeline_subobject_type::depth_stencil_state,   1, (void *)&dss },
        { pipeline_subobject_type::primitive_topology,    1, (void *)&topo },
        { pipeline_subobject_type::sample_mask,           1, (void *)&mask },
    };
    const bool ok = dev->create_pipeline(ds.cal_layout, 8, so, &ds.cal_pso);
    vs->Release(); ps->Release();
    if (!ok) { dev->destroy_pipeline_layout(ds.cal_layout); ds.cal_layout = { 0 }; return false; }

    resource_desc rd(kCalTiles, kCalTiles, 1, 1, format::r32g32_float, 1, memory_heap::default_,
                     resource_usage::render_target | resource_usage::copy_source);
    if (!dev->create_resource(rd, nullptr, resource_usage::copy_source, &ds.cal_rt) ||
        !dev->create_resource_view(ds.cal_rt, resource_usage::render_target,
                                   resource_view_desc(format::r32g32_float), &ds.cal_rtv))
    { Log("[depth-charge] could not create the calibration target"); return false; }

    resource_desc sd(kCalTiles, kCalTiles, 1, 1, format::r32g32_float, 1, memory_heap::readback,
                     resource_usage::copy_dest);
    if (!dev->create_resource(sd, nullptr, resource_usage::copy_dest, &ds.cal_stage))
    { Log("[depth-charge] could not create the calibration readback buffer"); return false; }

    Log("[depth-charge] calibration ready");
    return true;
}

// One frame of samples. Percentile across tiles, not the absolute extreme.
static void CalibrateFrame(device *dev, effect_runtime *runtime, device_state &ds, resource_view src_srv)
{
    if (!BuildCalibrator(dev, ds) || src_srv == 0) return;
    command_list *const cmd = runtime->get_command_queue()->get_immediate_command_list();
    if (cmd == nullptr) return;

    const resource res = ds.cal_rt;
    cmd->barrier(1, &res, &(const resource_usage &)resource_usage::copy_source,
                 &(const resource_usage &)resource_usage::render_target);
    cmd->bind_render_targets_and_depth_stencil(1, &ds.cal_rtv);
    const viewport vp = { 0.0f, 0.0f, (float)kCalTiles, (float)kCalTiles, 0.0f, 1.0f };
    cmd->bind_viewports(0, 1, &vp);
    const rect sc = { 0, 0, kCalTiles, kCalTiles };
    cmd->bind_scissor_rects(0, 1, &sc);
    cmd->bind_pipeline(pipeline_stage::all_graphics, ds.cal_pso);

    descriptor_table_update u = {};
    u.binding = 0; u.count = 1; u.type = descriptor_type::shader_resource_view; u.descriptors = &src_srv;
    cmd->push_descriptors(shader_stage::pixel, ds.cal_layout, 0, u);
    descriptor_table_update su = {};
    su.binding = 0; su.count = 1; su.type = descriptor_type::sampler; su.descriptors = &ds.convert_sampler;
    cmd->push_descriptors(shader_stage::pixel, ds.cal_layout, 1, su);
    cmd->draw(3, 1, 0, 0);

    cmd->barrier(1, &res, &(const resource_usage &)resource_usage::render_target,
                 &(const resource_usage &)resource_usage::copy_source);
    cmd->copy_texture_region(ds.cal_rt, 0, nullptr, ds.cal_stage, 0, nullptr);
    runtime->get_command_queue()->flush_immediate_command_list();

    subresource_data map = {};
    if (!dev->map_texture_region(ds.cal_stage, 0, nullptr, map_access::read_only, &map))
        return;

    std::vector<float> mins, maxs;
    mins.reserve(kCalTiles * kCalTiles); maxs.reserve(kCalTiles * kCalTiles);
    for (int y = 0; y < kCalTiles; y++)
    {
        const float *row = (const float *)((const uint8_t *)map.data + (size_t)y * map.row_pitch);
        for (int x = 0; x < kCalTiles; x++)
        {
            const float mn = row[x * 2 + 0], mx = row[x * 2 + 1];
            if (mx > 0.0f) { mins.push_back(mn); maxs.push_back(mx); }
        }
    }
    dev->unmap_texture_region(ds.cal_stage, 0);
    if (mins.size() < 8) return;                       // barely any geometry on screen

    // Which end of this buffer is "infinitely far"?
    //
    // Whatever value a buffer is CLEARED to means "nothing was drawn here", and
    // nothing-was-drawn is the far plane by definition. So the far end is whichever
    // extreme a large population of tiles share exactly - the sky. On PS2 and
    // GameCube that is the minimum; on DuckStation's reconstructed PS1 depth it is
    // the maximum, because the buffer is cleared to 1.0 and counts up with distance.
    // Deciding it per frame from the data means no per-emulator special case.
    {
        float gmax = 0.0f, gmin = 3.4e38f;
        for (float v : maxs) if (v > gmax) gmax = v;
        for (float v : mins) if (v < gmin) gmin = v;
        size_t at_max = 0, at_min = 0;
        for (float v : maxs) if (v >= gmax - 1e-4f) ++at_max;
        for (float v : mins) if (v <= gmin + 1e-4f) ++at_min;

        if (at_max > at_min && !g_cfg_linear_buffer)
        {
            // Mirror the frame: min and max swap roles as well as values.
            //
            // The clear value maps to exactly zero, and zero means "no surface here",
            // not "a surface at infinity". Those have to be DROPPED, not clamped to
            // something small - clamp them and the placeholder becomes the farthest
            // thing in the scene, which pins the far plane to a number the add-on
            // invented and crushes the whole picture into one flat value.
            const float eps = 1e-5f;
            std::vector<float> nmin, nmax;
            nmin.reserve(mins.size()); nmax.reserve(maxs.size());
            for (size_t i = 0; i < mins.size(); ++i)
            {
                const float a = 1.0f - maxs[i];        // farthest surface in the tile
                const float b = 1.0f - mins[i];        // nearest surface in the tile
                if (b <= eps) continue;                // nothing but sky here
                nmax.push_back(b);
                if (a > eps) nmin.push_back(a);        // sky in the tile: no far reading
            }
            mins.swap(nmin); maxs.swap(nmax);
            ++g_cal_flip_votes;
        }
        ++g_cal_vote_frames;
        if (mins.size() < 8 || maxs.size() < 8) return;
    }

    std::sort(mins.begin(), mins.end());
    std::sort(maxs.begin(), maxs.end());

    // The spatial distribution of one frame, logged once per run. The across-frames
    // quantiles cannot answer the question that matters - whether the far extreme is
    // a lone tile or a large population - because each frame contributes only its own
    // single extreme. This is the tile histogram inside a frame, which can.
    if (g_cal_frames == 0)
    {
        char line[640]; int n = 0;
        n += sprintf_s(line + n, sizeof(line) - n,
                       "[depth-charge] tile mins across one frame (log2 raw, %zu tiles):", mins.size());
        for (int i = 0; i <= 20; ++i)
        {
            const size_t idx = (mins.size() - 1) * i / 20;
            n += sprintf_s(line + n, sizeof(line) - n, " %.2f", log2f(mins[idx]));
        }
        Log("%s", line);
        n = 0;
        n += sprintf_s(line + n, sizeof(line) - n,
                       "[depth-charge] tile maxs across one frame (log2 raw):");
        for (int i = 0; i <= 20; ++i)
        {
            const size_t idx = (maxs.size() - 1) * i / 20;
            n += sprintf_s(line + n, sizeof(line) - n, " %.2f", log2f(maxs[idx]));
        }
        Log("%s", line);
    }


    // Far-end rejection, the mirror of the overlay rejection above.
    //
    // A backdrop - skybox, distant water plate, matte - sits at a fixed enormous
    // distance with NOTHING between it and the real scenery. Real scenery is
    // continuous, so its sorted depths step smoothly; a backdrop announces itself as a
    // single large multiplicative gap. Anchoring the far end on it stretches the scale
    // by that whole gap and crushes everything a player actually walks on into the
    // bottom fraction of the range.
    //
    // So: find the largest octave gap in the lower half and, if it is decisive, start
    // above it. Bounded to the lower half so this can never reject the scene itself,
    // and a compressed buffer (THUG2 spans 1.5% end to end) has no such gap at all,
    // which makes this a no-op there rather than a special case.
    // Two sizes of gap, because there are two different things down there.
    //
    // A backdrop is part of the picture - a few octaves past the scenery, a minority
    // of the frame - so rejecting it is bounded to the lower half, or a legitimate
    // long view could be mistaken for one.
    //
    // An EMPTY background is not part of the picture at all. Dolphin leaves two thirds
    // of the frame at 2^-24, the smallest value a 24-bit Z can hold, eighteen octaves
    // below anything real. Nothing in a rendered scene is eighteen octaves from its
    // neighbour, so a gap that size is not a judgement call and does not need the
    // safety bound - it can be trusted to well past the halfway mark.
    size_t farIdx = 0;
    {
        const size_t n = mins.size();
        const size_t limit = n - n / 20;           // never discard the last 5%
        float bestGap = 0.0f; size_t bestIdx = 0;
        for (size_t i = 0; i + 1 < limit; ++i)
        {
            const float gap = log2f(mins[i + 1]) - log2f(mins[i]);
            if (gap > bestGap) { bestGap = gap; bestIdx = i + 1; }
        }
        const bool backdrop = (bestGap >= 2.0f && bestIdx < n / 2);   // in the picture
        const bool empty    = (bestGap >= 6.0f);                      // not in the picture
        if (backdrop || empty)
        {
            farIdx = bestIdx;
            if (g_cal_frames == 0)
                Log("[depth-charge] %s rejected: %zu of %zu tiles below a %.1f-octave gap "
                    "(far end moves %.2f -> %.2f)", empty ? "empty background" : "backdrop",
                    bestIdx, n, bestGap, log2f(mins.front()), log2f(mins[farIdx]));
        }
    }

    // Everything below that floor is not geometry, so it must not vote on the near end
    // either. Left in, it dominates the median the overlay test is built on, and real
    // geometry - a million times larger here - gets thrown out as if it were a HUD.
    if (farIdx > 0)
    {
        const float floor = mins[farIdx] * 0.5f;
        std::vector<float> live;
        live.reserve(maxs.size());
        for (float v : maxs) if (v > floor) live.push_back(v);
        if (live.size() >= 4) maxs.swap(live);
    }

    // 2D overlays are drawn at the nearest possible Z, so they sit far above the
    // geometry and would otherwise BE the "nearest surface" - which is what threw
    // the first run: it read the HUD at the format maximum and scaled everything to
    // it. Reject anything well above the median before taking the near end. Real
    // geometry stays within a modest factor of the median; an overlay spike does not.
    const float medMax = maxs[maxs.size() / 2];
    std::vector<float> geo;
    geo.reserve(maxs.size());
    for (float v : maxs)
        if (v <= medMax * 1.5f) geo.push_back(v);
    if (geo.empty()) geo.push_back(medMax);

    // The values just rejected are the overlay. Keep the largest - the HUD is drawn
    // at the nearest Z the format can hold, so nothing sits above it - and it becomes
    // the second half of the cutoff derivation once A and F are known.
    if (maxs.back() > medMax * 1.5f)
        g_cal_huds.push_back(maxs.back());


    // A linear buffer runs the other way round, so none of the filtering above applies
    // to it: the LARGEST values are the most distant geometry, not the HUD, and the
    // smallest are the nearest surface, not the sky. Rejecting the top 1.5x-of-median
    // as overlay therefore throws the far end away and anchors the scale far too
    // close - which saturates everything past mid-distance to the far plane. Take the
    // honest extremes here and let the across-frames percentile reject outliers.
    if (g_cfg_linear_buffer)
    {
        // 1.0 is the clear value: "nothing was drawn here", i.e. the far plane itself
        // rather than a surface at it. Anchoring to that pins the scale to infinity and
        // collapses the whole scene into the near end. Walk down to the most distant
        // value that is actually geometry.
        size_t hi = maxs.size();
        while (hi > 0 && maxs[hi - 1] >= 0.999f) --hi;
        if (hi == 0) return;                                   // nothing but sky

        // A small percentile in from each end, so one stray tile cannot set the range.
        const size_t fi = (hi > 20) ? (hi - 1 - hi / 20) : (hi - 1);
        const size_t ni = mins.size() / 20;

        g_cal_maxs.push_back(maxs[fi]);                        // farthest real surface
        g_cal_mins.push_back(mins[ni]);                        // nearest real surface
        g_cal_frames++;
        return;
    }

    g_cal_mins.push_back(mins[farIdx]);                        // farthest tile this frame
    g_cal_maxs.push_back(geo.back());                          // nearest tile, overlays removed
    g_cal_frames++;
}

static void CalibrateFinish()
{
    g_cal_state = CAL_SHOWING;
    g_cal_started = (double)GetTickCount64() / 1000.0;

    // The window is short and the readback is synchronous, so a game running at
    // 30fps or hitching gives fewer frames than the arithmetic suggests. Keep the
    // floor low enough that a slow emulator still clears it.
    if (g_cal_mins.size() < 15)
    {
        g_cal_ok = false;
        sprintf_s(g_cal_status, "Not enough usable frames (%zu). Try again while looking at the scene.",
                  g_cal_mins.size());
        Log("[depth-charge] calibration failed: only %zu frames had geometry", g_cal_mins.size());
        return;
    }

    std::sort(g_cal_mins.begin(), g_cal_mins.end());
    std::sort(g_cal_maxs.begin(), g_cal_maxs.end());

    // A linear buffer has nothing to solve but its scale. There is no asymptote to
    // subtract - the value IS the distance - so the only question is which distance
    // should read as 1.0. Take the farthest surface and put it at 0.95, same anchor
    // the 1/z path uses, and leave the shape completely alone.
    if (g_cfg_linear_buffer)
    {
        // Without the flip, LARGE is far. The far end therefore comes from the maxima.
        // Per-frame values are already trimmed; take the median across frames so a
        // single bad frame - a cut scene, a door opening - cannot set the scale.
        const float farRawL  = g_cal_maxs[g_cal_maxs.size() / 2];
        const float nearRawL = g_cal_mins[g_cal_mins.size() / 2];
        if (farRawL <= 0.0f)
        {
            g_cal_ok = false;
            sprintf_s(g_cal_status, "The buffer reads zero everywhere - nothing to scale.");
            Log("[depth-charge] linear calibration failed: far raw %.9f", farRawL);
            return;
        }
        // Report what was actually measured, so the greyed line below the sliders is
        // not showing leftovers from a previous run in the other mode.
        g_cal_raw_far = farRawL; g_cal_raw_near = nearRawL;
        g_cal_result_zero = 0.0f;                 // no asymptote in a linear buffer
        g_cal_result_far  = 0.95f / farRawL;
        g_cfg_zero_log2   = g_cal_result_zero;
        g_cfg_far_scale   = g_cal_result_far;
        g_cfg_flip_depth  = false;
        g_cfg_overlay_cutoff = 0.0f;
        g_cutoff_tries = 1200;
        g_cal_ok = true;
        MarkUserProfile();
        sprintf_s(g_cal_status,
                  "Linear buffer: scale %.1f  (from %zu frames)\n"
                  "No asymptote to remove - the buffer already holds distance.",
                  g_cal_result_far, g_cal_mins.size());
        Log("[depth-charge] calibrated LINEAR buffer: far_scale=%.1f "
            "(far raw %.9f, near raw %.9f, from %zu frames)",
            g_cal_result_far, farRawL, nearRawL, g_cal_mins.size());
        WriteConfig();
        return;
    }

    // The shape of the distribution, not just its ends. Two very different scenes
    // produce the same pair of extremes: one where depth runs smoothly out to the
    // horizon, and one where a skybox sits alone thousands of units past everything
    // else. Only the second wants its far end rejected, and the extremes alone
    // cannot tell them apart - a cliff between quantiles can.
    {
        char line[512]; int n = 0;
        n += sprintf_s(line + n, sizeof(line) - n, "[depth-charge] far-end quantiles (log2 raw):");
        for (int i = 0; i <= 10; ++i)
        {
            const size_t idx = (g_cal_mins.size() - 1) * i / 10;
            n += sprintf_s(line + n, sizeof(line) - n, " %.3f", log2f(g_cal_mins[idx]));
        }
        Log("%s", line);
        n = 0;
        n += sprintf_s(line + n, sizeof(line) - n, "[depth-charge] near-end quantiles (log2 raw):");
        for (int i = 0; i <= 10; ++i)
        {
            const size_t idx = (g_cal_maxs.size() - 1) * i / 10;
            n += sprintf_s(line + n, sizeof(line) - n, " %.3f", log2f(g_cal_maxs[idx]));
        }
        Log("%s", line);
    }

    // The farthest and nearest raw values actually observed. Low/high percentiles
    // rather than absolute extremes, so a freak frame cannot set either end.
    const float farRaw  = g_cal_mins[g_cal_mins.size() *  2 / 100];
    const float nearRaw = g_cal_maxs[g_cal_maxs.size() * 98 / 100];
    g_cal_raw_far = farRaw; g_cal_raw_near = nearRaw;

    if (farRaw <= 0.0f || nearRaw <= farRaw)
    {
        g_cal_ok = false;
        sprintf_s(g_cal_status, "Could not separate near from far - try again facing more of the scene.");
        Log("[depth-charge] calibration rejected: far=%.9f near=%.9f", farRaw, nearRaw);
        return;
    }

    // Solve BOTH constants from those two points.
    //
    // The asymptote is the value at infinite distance, so it lies BELOW everything
    // ever observed - reading it off the farthest visible surface always lands too
    // high, and too high is precisely the error that saturates the far field. Instead,
    // require the observed range to land on a chosen span of the output:
    //
    //     d = 1 / ((raw - A) * F),   d(nearRaw) = kNear,  d(farRaw) = kFar
    //
    // Two equations, two unknowns. A falls out below the farthest surface on its own,
    // which is where it belongs.
    const float kNear = 0.05f;      // nearest surface lands here
    const float kFar  = 0.95f;      // farthest surface lands here, short of saturating
    const float r = kFar / kNear;

    // Two regimes, and the buffer itself says which one it is.
    //
    // THUG2's is compressed: near and far differ by 7%, so almost all of the value
    // is a constant and A has to be dug out from under it. But a buffer whose
    // observed range already spans more than r is an ordinary one - there is no
    // large constant hiding beneath it, and solving as if there were drives A
    // negative, because no positive A can shrink a 25x range down to 19x.
    //
    // That is not a failure to detect, it is the answer: A is zero, or near enough
    // that nothing in the sample argues otherwise. Any positive A would only
    // compress the near field further on no evidence.
    const bool wide = (nearRaw >= farRaw * r);
    float A = wide ? farRaw * 0.0009765625f            // 2^-10 of the farthest: negligible
                   : (r * farRaw - nearRaw) / (r - 1.0f);
    if (A <= 0.0f || nearRaw - A <= 0.0f)
    {
        g_cal_ok = false;
        sprintf_s(g_cal_status, "This depth buffer does not fit the expected shape - set the sliders by hand.");
        Log("[depth-charge] calibration rejected: solved A=%.9f from far=%.9f near=%.9f", A, farRaw, nearRaw);
        return;
    }

    float zero = log2f(A);

    // Snap to a whole power of two when we land very close to one.
    //
    // Console games build their Z scale out of integer ranges, so the asymptote
    // tends to sit exactly on a power of two rather than near one - THUG2's is
    // 2^-9, which in 16-bit Z units is exactly half the range. The solve above
    // only ever sees the geometry the player happened to look at, so it lands
    // near the truth rather than on it. If the answer is within a couple of
    // percent of a power of two, the power of two is almost certainly the real
    // number and the remainder is sampling error.
    //
    // This matters more than it looks: F is a reciprocal of (nearRaw - A), a
    // small difference between two large numbers, so a 0.5% error in A moves F
    // by a factor of several. Fixing A exactly is what makes F reproducible.
    const float snapped = roundf(zero);
    const float snappedA = exp2f(snapped);
    if (!wide && g_cfg_snap_zero && fabsf(zero - snapped) < 0.03f && snappedA < farRaw * 0.999f && snappedA > 0.0f)
    {
        Log("[depth-charge] snapping zero point %.4f -> %.4f (power of two)", zero, snapped);
        zero = snapped;
        A = snappedA;
    }

    // Anchor the scale on the FAR point, not the near one. With the unsnapped A the
    // two are identical by construction, but once A moves the near anchor lets the
    // far end run past 1.0 and saturate - exactly the flat red plateau this whole
    // pass exists to avoid. Anchoring far can only compress the near field toward
    // zero instead, which is graceful.
    g_cal_result_zero = zero;
    g_cal_result_far  = (1.0f / kFar) / (farRaw - A);
    g_cfg_zero_log2 = g_cal_result_zero;
    g_cfg_far_scale = g_cal_result_far;

    // and, from the same sample set, where the overlay stops and geometry starts.
    //
    // Both raw values are already in hand: the HUD cluster rejected each frame, and
    // the nearest surviving geometry. Converting both through the constants just
    // solved puts them on the 0..1 depth scale every shader reads.
    //
    // The cutoff goes midway between them on a log scale, so the margin is
    // proportional on both sides rather than favouring whichever end is smaller.
    g_cal_hud_depth = g_cal_geo_depth = 0.0f;
    bool cutoff_derived = false;
    if (g_cal_huds.size() >= 5)
    {
        std::sort(g_cal_huds.begin(), g_cal_huds.end());
        const float hudRaw = g_cal_huds[g_cal_huds.size() / 2];
        if (hudRaw > nearRaw && hudRaw - A > 0.0f && nearRaw - A > 0.0f)
        {
            const float dHud = 1.0f / ((hudRaw  - A) * g_cal_result_far);
            const float dGeo = 1.0f / ((nearRaw - A) * g_cal_result_far);
            // Refuse when the two are not cleanly separated - a cutoff placed inside
            // real geometry deletes the ground, which is far worse than a shaded HUD.
            if (dGeo > dHud * 4.0f)
            {
                float cut = sqrtf(dHud * dGeo);
                if (cut > 0.2f) cut = 0.2f;             // the slider's own range
                g_cal_hud_depth = dHud;
                g_cal_geo_depth = dGeo;
                g_cfg_overlay_cutoff = cut;
                cutoff_derived = true;
                Log("[depth-charge] overlay cutoff %.5f (HUD at %.5f, nearest geometry at %.5f, "
                    "%zu overlay samples)", cut, dHud, dGeo, g_cal_huds.size());
            }
            else
            {
                Log("[depth-charge] overlay cutoff not derived: HUD %.5f and geometry %.5f "
                    "too close to separate", dHud, dGeo);
            }
        }
    }

    // Whether or not one was derived, say so out loud. Leaving the shader holding a
    // cutoff measured in a DIFFERENT game is worse than having none: the number is in
    // depth units, so it means something else entirely here, and the symptom is real
    // geometry silently disappearing. No answer pushes zero - exclusion off.
    if (!cutoff_derived)
    {
        g_cfg_overlay_cutoff = 0.0f;
        Log("[depth-charge] overlay cutoff cleared (%zu overlay samples) - HUD exclusion off",
            g_cal_huds.size());
    }
    g_cutoff_tries = 1200;                              // ~20s of frames to find the shader

    // A majority of frames agreeing is enough; a stray frame full of sky is not.
    g_cfg_flip_depth = (g_cal_vote_frames > 0 && g_cal_flip_votes * 2 > g_cal_vote_frames);
    if (g_cfg_flip_depth)
        Log("[depth-charge] buffer counts 0 at the eye and 1 at infinity "
            "(%d of %d frames) - flipping it to match everything else",
            g_cal_flip_votes, g_cal_vote_frames);

    g_cal_ok = true;
    MarkUserProfile();
    const char *shape = wide ? "  [wide-range buffer - no offset to remove]" : "";
    if (cutoff_derived)
        sprintf_s(g_cal_status, "Zero point %.4f, far distance %.0f, HUD cutoff %.4f  (from %zu frames)%s",
                  g_cal_result_zero, g_cal_result_far, g_cfg_overlay_cutoff, g_cal_mins.size(), shape);
    else
        sprintf_s(g_cal_status, "Zero point %.4f, far distance %.0f  (from %zu frames)%s\n"
                  "Could not separate the HUD from geometry here, so HUD exclusion is off.",
                  g_cal_result_zero, g_cal_result_far, g_cal_mins.size(), shape);
    Log("[depth-charge] calibrated: zero_point_log2=%.4f far_scale=%.1f "
        "(far raw %.9f, near raw %.9f, from %zu frames)",
        g_cal_result_zero, g_cal_result_far, farRaw, nearRaw, g_cal_mins.size());
    WriteConfig();
}

// (Oct 23 2026) ReShade's own depth settings, as ReShade reports them, so the published depth can be
// written the way they will read it back: far plane, reversed, multiplier, logarithmic, upside down,
// mirrored. Until ReShade has been asked, its own defaults (ReShade.fxh).
static float g_enc[6] = { 1000.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f };

// stage 3: linearise
//
// A small fullscreen pass owned by the add-on. Runs on the placed depth texture
// and writes true linear view-space depth into an R32_FLOAT target, which is what
// gets published under the DEPTH semantic.

static const char *kConvertHLSL = R"__(
Texture2D<float>  src : register(t0);
SamplerState      smp : register(s0);
cbuffer Params : register(b0)
{
    float zero_point;   // A: the value the buffer asymptotes to at infinity
    float far_scale;    // which distance reads as 1.0
    float flip;         // 1 = buffer counts 0 at the eye, 1 at infinity
    float curve;        // 1 = linear in distance; >1 leans back toward the raw curve
    float already_linear;   // 1 = raw is distance already: scale it, never invert it
    float upside_down;      // (Oct 23 2026) ReShade's RESHADE_DEPTH_INPUT_IS_UPSIDE_DOWN and
    float mirrored;         // RESHADE_DEPTH_INPUT_IS_MIRRORED: written the way round ReShade reads it
    float pad2;
    float4 place;       // (Oct 19 2026) published uv -> capture uv: uv * place.xy + place.zw (identity: 1, 1, 0, 0)
    float4 box;         // the captured rectangle in capture uv (u0, v0, u1, v1): outside it nothing was drawn
    float4 enc;         // (Oct 23 2026) ReShade's depth settings: far plane, reversed, multiplier, logarithmic
};

void vs_main(uint id : SV_VertexID, out float4 pos : SV_Position, out float2 uv : TEXCOORD)
{
    uv  = float2((id == 2) ? 2.0 : 0.0, (id == 1) ? 2.0 : 0.0);
    pos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
}

float Distance(float2 uv)
{
    // (Oct 19 2026) Where this texel of the published depth is in the capture: the picture's place on a screen of another
    // shape (the bars beside a 16:9 picture on an ultrawide). Outside the captured rectangle nothing was drawn: far plane.
    const float2 suv = uv * place.xy + place.zw;
    if (any(suv < box.xy) || any(suv > box.zw))
        return 1.0;
    float raw = src.SampleLevel(smp, suv, 0);

    // A buffer that already holds distance needs a scale and nothing else. No
    // asymptote to remove, no reciprocal, no flip - it is small near and large far,
    // which is what every effect wants. Inverting it would be actively destructive.
    if (already_linear > 0.5)
    {
        if (raw <= 0.0) return 1.0;                 // nothing drawn: far plane
        float d = saturate(raw * far_scale);
        return (curve == 1.0) ? d : pow(d, curve);
    }

    // Put an eye-at-zero buffer into the same orientation as every other one. The
    // clear value (1.0) becomes 0, which the sky test below then reads correctly.
    if (flip > 0.5)
        raw = 1.0 - raw;

    // Nothing was drawn here (sky): treat as the far plane.
    if (raw <= 0.0)
        return 1.0;

    // Remove the asymptote, then reciprocal. Anything at or past it is at
    // infinity as far as this buffer can express, so it pins far.
    float varying = raw - zero_point;
    if (varying <= 0.0)
        return 1.0;

    // NB: not named "linear" - that is a reserved interpolation modifier in HLSL.
    float dist = saturate(1.0 / (varying * far_scale));

    // Reshape without changing either end: 0 stays 0 and 1 stays 1, so the near and
    // far anchors the calibration solved for are untouched. Only what happens between
    // them moves, which is exactly what separates true distance from the buffer's own
    // hyperbolic curve.
    return (curve == 1.0) ? dist : pow(dist, curve);
}

// (Oct 23 2026) Written the way ReShade's own depth settings will read it back. ReShade's
// GetLinearizedDepth() multiplies, undoes a logarithmic buffer, flips a reversed one, then divides
// by FAR - depth * (FAR - 1). This is that, run backwards, so whatever the settings are, an effect
// reads the distance above. With the neutral settings Depth Charge used to ask for (far plane 1,
// not reversed, multiplier 1, not logarithmic) it is the distance itself, unchanged.
float Encode(float dist)
{
    const float F = enc.x;
    float d = (F == 1.0) ? dist : dist * F / (1.0 + dist * (F - 1.0));
    if (enc.y > 0.5) d = 1.0 - d;
    if (enc.w > 0.5) { const float C = 0.01; d = log(d * C + 1.0) / log(C + 1.0); }
    return d / enc.z;
}

float ps_main(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    // ReShade flips its read when told the depth is upside down or mirrored, so write it flipped.
    if (upside_down > 0.5) uv.y = 1.0 - uv.y;
    if (mirrored    > 0.5) uv.x = 1.0 - uv.x;
    return Encode(Distance(uv));
}
)__";

static bool BuildConverter(device *dev, device_state &ds)
{
    if (ds.convert_ready)  return true;
    if (ds.convert_tried)  return false;
    ds.convert_tried = true;

    const device_api api = dev->get_api();
    if (api != device_api::d3d10 && api != device_api::d3d11 && api != device_api::d3d12)
    {
        Log("[depth-charge] linearisation needs Direct3D 10/11/12; this device is something else, "
            "publishing raw depth instead");
        return false;
    }

    ID3DBlob *vs = nullptr, *ps = nullptr, *err = nullptr;
    const UINT flags = D3DCOMPILE_OPTIMIZATION_LEVEL3;
    if (FAILED(D3DCompile(kConvertHLSL, strlen(kConvertHLSL), nullptr, nullptr, nullptr,
                          "vs_main", "vs_5_0", flags, 0, &vs, &err)) ||
        FAILED(D3DCompile(kConvertHLSL, strlen(kConvertHLSL), nullptr, nullptr, nullptr,
                          "ps_main", "ps_5_0", flags, 0, &ps, &err)))
    {
        Log("[depth-charge] could not compile the linearisation shader: %s",
            err ? (const char *)err->GetBufferPointer() : "(no message)");
        if (vs)  vs->Release();
        if (ps)  ps->Release();
        if (err) err->Release();
        return false;
    }
    if (err) { err->Release(); err = nullptr; }

    // t0 (the placed depth), s0 (point sampler), b0 (the two constants)
    descriptor_range srv_range = {};
    srv_range.binding = 0; srv_range.dx_register_index = 0; srv_range.count = 1;
    srv_range.type = descriptor_type::shader_resource_view;
    srv_range.visibility = shader_stage::pixel;

    descriptor_range smp_range = {};
    smp_range.binding = 0; smp_range.dx_register_index = 0; smp_range.count = 1;
    smp_range.type = descriptor_type::sampler;
    smp_range.visibility = shader_stage::pixel;

    constant_range cb_range = {};
    cb_range.binding = 0; cb_range.dx_register_index = 0; cb_range.count = 20; // (Oct 19 2026: + place, box; Oct 23: + enc)
    cb_range.visibility = shader_stage::pixel;

    const pipeline_layout_param params[3] = {
        pipeline_layout_param(srv_range),
        pipeline_layout_param(smp_range),
        pipeline_layout_param(cb_range),
    };
    if (!dev->create_pipeline_layout(3, params, &ds.convert_layout))
    {
        Log("[depth-charge] could not create the linearisation pipeline layout");
        vs->Release(); ps->Release();
        return false;
    }

    shader_desc vs_desc = {}; vs_desc.code = vs->GetBufferPointer(); vs_desc.code_size = vs->GetBufferSize();
    shader_desc ps_desc = {}; ps_desc.code = ps->GetBufferPointer(); ps_desc.code_size = ps->GetBufferSize();

    const format rt_format = format::r32_float;
    blend_desc         blend = {};
    rasterizer_desc    raster = {};   raster.cull_mode = cull_mode::none;
    depth_stencil_desc depth = {};    depth.depth_enable = false; depth.stencil_enable = false;
    const primitive_topology topology = primitive_topology::triangle_list;
    const uint32_t sample_mask = 0xFFFFFFFF;
    const uint32_t rt_count = 1;

    const pipeline_subobject subobjects[] = {
        { pipeline_subobject_type::vertex_shader,           1, (void *)&vs_desc  },
        { pipeline_subobject_type::pixel_shader,            1, (void *)&ps_desc  },
        { pipeline_subobject_type::render_target_formats,   1, (void *)&rt_format },
        { pipeline_subobject_type::blend_state,             1, (void *)&blend    },
        { pipeline_subobject_type::rasterizer_state,        1, (void *)&raster   },
        { pipeline_subobject_type::depth_stencil_state,     1, (void *)&depth    },
        { pipeline_subobject_type::primitive_topology,      1, (void *)&topology },
        { pipeline_subobject_type::sample_mask,             1, (void *)&sample_mask },
    };
    (void)rt_count;

    const bool ok = dev->create_pipeline(ds.convert_layout,
                                         (uint32_t)(sizeof(subobjects) / sizeof(subobjects[0])),
                                         subobjects, &ds.convert_pso);
    vs->Release(); ps->Release();
    if (!ok)
    {
        Log("[depth-charge] could not create the linearisation pipeline");
        dev->destroy_pipeline_layout(ds.convert_layout); ds.convert_layout = { 0 };
        return false;
    }

    sampler_desc sd = {};
    sd.filter = filter_mode::min_mag_mip_point;
    sd.address_u = sd.address_v = sd.address_w = texture_address_mode::clamp;
    if (!dev->create_sampler(sd, &ds.convert_sampler))
    {
        Log("[depth-charge] could not create the linearisation sampler");
        dev->destroy_pipeline(ds.convert_pso); ds.convert_pso = { 0 };
        dev->destroy_pipeline_layout(ds.convert_layout); ds.convert_layout = { 0 };
        return false;
    }

    ds.convert_ready = true;
    Log("[depth-charge] linearisation pass ready - publishing true linear depth, written for "
        "ReShade's own depth settings so effects read it back as distance");
    return true;
}

// Returns the SRV to publish: the linearised one when it worked, else the input.
// place (Oct 19 2026, optional): 8 floats - published uv -> capture uv scale (2) and offset (2), then the captured rectangle
// in capture uv (u0, v0, u1, v1); outside it the published depth is "nothing drawn". Null: the capture as it is.
static resource_view Linearise(device *dev, effect_runtime *runtime, device_state &ds,
                               resource src, resource_view src_srv, uint32_t w, uint32_t h,
                               const float *place = nullptr)
{
    if (!g_cfg_linearise || src == 0 || src_srv == 0)
        return src_srv;

    // Calibration samples the RAW depth, before conversion - it is solving for the
    // constants the conversion needs, so it cannot use the converted result.
    if (g_cal_state == CAL_CAPTURING)
    {
        const double now = (double)GetTickCount64() / 1000.0;
        if (now - g_cal_started >= g_cal_seconds)
            CalibrateFinish();
        else
            CalibrateFrame(dev, runtime, ds, src_srv);
    }
    else if (g_cal_state == CAL_SHOWING)
    {
        const double now = (double)GetTickCount64() / 1000.0;
        if (now - g_cal_started >= 3.0)
        {
            g_cal_state = CAL_IDLE;
            g_dbg_view  = g_dbg_view_before;    // put the view back where it was
        }
    }

    if (!BuildConverter(dev, ds))
    {
        if (!ds.convert_warned)
        {
            ds.convert_warned = true;
            Log("[depth-charge] falling back to raw depth; a shader-side conversion is still needed");
        }
        return src_srv;
    }

    if (ds.lin == 0 || ds.lin_w != w || ds.lin_h != h)
    {
        if (ds.lin_rtv != 0) { dev->destroy_resource_view(ds.lin_rtv); ds.lin_rtv = { 0 }; }
        if (ds.lin_srv != 0) { dev->destroy_resource_view(ds.lin_srv); ds.lin_srv = { 0 }; }
        if (ds.lin     != 0) { dev->destroy_resource(ds.lin);          ds.lin     = { 0 }; }

        resource_desc rd(w, h, 1, 1, format::r32_float, 1, memory_heap::default_,
                         resource_usage::render_target | resource_usage::shader_resource);
        if (!dev->create_resource(rd, nullptr, resource_usage::shader_resource, &ds.lin))
        {
            Log("[depth-charge] could not create the %ux%u linear depth texture", w, h);
            ds.lin = { 0 }; return src_srv;
        }
        if (!dev->create_resource_view(ds.lin, resource_usage::render_target,
                resource_view_desc(format::r32_float), &ds.lin_rtv) ||
            !dev->create_resource_view(ds.lin, resource_usage::shader_resource,
                resource_view_desc(format::r32_float), &ds.lin_srv))
        {
            Log("[depth-charge] could not create views for the linear depth texture");
            if (ds.lin_rtv != 0) { dev->destroy_resource_view(ds.lin_rtv); ds.lin_rtv = { 0 }; }
            dev->destroy_resource(ds.lin); ds.lin = { 0 };
            return src_srv;
        }
        ds.lin_w = w; ds.lin_h = h;
        Log("[depth-charge] linear depth target %ux%u", w, h);
    }

    command_list *const cmd = runtime->get_command_queue()->get_immediate_command_list();
    if (cmd == nullptr)
        return src_srv;

    const resource       res [2] = { src, ds.lin };
    const resource_usage from[2] = { resource_usage::shader_resource, resource_usage::shader_resource };
    const resource_usage to  [2] = { resource_usage::shader_resource, resource_usage::render_target  };
    cmd->barrier(2, res, from, to);

    cmd->bind_render_targets_and_depth_stencil(1, &ds.lin_rtv);

    const viewport vp = { 0.0f, 0.0f, (float)w, (float)h, 0.0f, 1.0f };
    cmd->bind_viewports(0, 1, &vp);
    const rect sc = { 0, 0, (int32_t)w, (int32_t)h };
    cmd->bind_scissor_rects(0, 1, &sc);

    cmd->bind_pipeline(pipeline_stage::all_graphics, ds.convert_pso);

    descriptor_table_update srv_update = {};
    srv_update.binding = 0; srv_update.count = 1;
    srv_update.type = descriptor_type::shader_resource_view;
    srv_update.descriptors = &src_srv;
    cmd->push_descriptors(shader_stage::pixel, ds.convert_layout, 0, srv_update);

    descriptor_table_update smp_update = {};
    smp_update.binding = 0; smp_update.count = 1;
    smp_update.type = descriptor_type::sampler;
    smp_update.descriptors = &ds.convert_sampler;
    cmd->push_descriptors(shader_stage::pixel, ds.convert_layout, 1, smp_update);

    static const float kIdentity[8] = { 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f };
    const float *const pl = place ? place : kIdentity;
    const float constants[20] = { exp2f(g_cfg_zero_log2), g_cfg_far_scale,
                                  g_cfg_flip_depth ? 1.0f : 0.0f, g_cfg_depth_curve,
                                  g_cfg_linear_buffer ? 1.0f : 0.0f, g_enc[4], g_enc[5], 0.0f,
                                  pl[0], pl[1], pl[2], pl[3], pl[4], pl[5], pl[6], pl[7],
                                  g_enc[0], g_enc[1], g_enc[2], g_enc[3] };
    cmd->push_constants(shader_stage::pixel, ds.convert_layout, 2, 0, 20, constants);

    cmd->draw(3, 1, 0, 0);

    cmd->barrier(2, res, to, from);
    return ds.lin_srv;
}

// the inspector
//
// Debug views, drawn over the finished frame. Everything is derived from the
// depth buffer alone, normals included, so none of it depends on any particular
// shader pack being installed.

static const char *kDebugHLSL = R"__(
Texture2D<float>  rawTex : register(t0);   // depth as captured
Texture2D<float>  linTex : register(t1);   // depth after linearisation
SamplerState      smp    : register(s0);
cbuffer Params : register(b0)
{
    float view;        // which view
    float log_min;     // log2 of raw, far end
    float log_max;     // log2 of raw, near end
    float near_gain;
    float bands;
    float hud_cut;
    float tan_half_fov;
    float aspect;
    float fade;        // 0 = invisible, 1 = fully replacing the frame
    float upside_down, mirrored, pad2;
    float4 enc;        // (Oct 23 2026) ReShade's depth settings: far plane, reversed, multiplier, logarithmic
};

void vs_main(uint id : SV_VertexID, out float4 pos : SV_Position, out float2 uv : TEXCOORD)
{
    uv  = float2((id == 2) ? 2.0 : 0.0, (id == 1) ? 2.0 : 0.0);
    pos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
}

float3 Spectrum(float t)
{
    t = saturate(t);
    if (t < 0.25) return lerp(float3(0.10,0.00,0.35), float3(0.00,0.40,1.00), t * 4.0);
    if (t < 0.50) return lerp(float3(0.00,0.40,1.00), float3(0.00,0.95,0.55), (t - 0.25) * 4.0);
    if (t < 0.75) return lerp(float3(0.00,0.95,0.55), float3(1.00,0.90,0.10), (t - 0.50) * 4.0);
                  return lerp(float3(1.00,0.90,0.10), float3(1.00,0.10,0.10), (t - 0.75) * 4.0);
}

// (Oct 23 2026) The published depth is written the way ReShade's own depth settings read it (see
// Encode in the conversion pass). Read it back the same way, so these views show the distance.
float Lin(float2 uv)
{
    if (upside_down > 0.5) uv.y = 1.0 - uv.y;
    if (mirrored    > 0.5) uv.x = 1.0 - uv.x;
    float depth = linTex.SampleLevel(smp, uv, 0) * enc.z;
    if (enc.w > 0.5) { const float C = 0.01; depth = (exp(depth * log(C + 1.0)) - 1.0) / C; }
    if (enc.y > 0.5) depth = 1.0 - depth;
    return depth / (enc.x - depth * (enc.x - 1.0));
}

// View-space position from a linear depth. Only the ratio matters for normals,
// so an approximate field of view is good enough to see the geometry.
float3 ViewPos(float2 uv, float d, float2 texel)
{
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return float3(ndc.x * tan_half_fov * aspect * d, ndc.y * tan_half_fov * d, d);
}

float4 ps_main(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float2 texel;
    linTex.GetDimensions(texel.x, texel.y);
    texel = 1.0 / max(texel, 1.0);

    float raw = rawTex.SampleLevel(smp, uv, 0);
    float d   = Lin(uv);

    // 1: linear depth, grey
    if (view < 1.5)
        return float4(d.xxx, fade);

    // 2: linear depth, false colour - a correct curve sweeps the whole spectrum
    if (view < 2.5)
        return float4(Spectrum(d), fade);

    // 3: RAW on a log scale. A 1/distance buffer spans decades; a linear ramp
    //    piles it all into one bucket, which is what a flat plateau looks like.
    if (view < 3.5)
    {
        if (raw <= 0.0) return float4(1.0, 0.0, 1.0, fade);
        float t = saturate((log2(raw) - log_min) / max(log_max - log_min, 0.001));
        return float4(Spectrum(t), fade);
    }

    // 4: near-field zoom. Magenta means exactly zero, not merely small.
    if (view < 4.5)
    {
        if (d <= 0.0) return float4(1.0, 0.0, 1.0, fade);
        return float4(saturate(d * near_gain).xxx, fade);
    }

    // 5: quantisation bands - wide bands mean coarse depth
    if (view < 5.5)
        return float4(frac(raw * bands).xxx, fade);

    // 6/7: normals reconstructed from depth, and where that reconstruction fails
    float3 pC = ViewPos(uv,                          d,                                          texel);
    float3 pR = ViewPos(uv + float2(texel.x, 0), Lin(uv + float2(texel.x, 0)), texel);
    float3 pB = ViewPos(uv + float2(0, texel.y), Lin(uv + float2(0, texel.y)), texel);
    float3 n  = normalize(cross(pR - pC, pB - pC));

    if (view < 6.5)
        return float4(n * 0.5 + 0.5, fade);

    if (view < 7.5)
    {
        // red where the normal has collapsed to facing straight at the camera,
        // which is what a degenerate reconstruction produces
        float flat_ = saturate((abs(n.z) - 0.985) / 0.015);
        return float4(lerp(float3(0.05, 0.05, 0.10), float3(1.0, 0.0, 0.0), flat_), fade);
    }

    // 8: overlay rejection band - 2D sprites drawn at near Z
    float3 base = Spectrum(d) * 0.35;
    bool rejected = (d > 0.0 && d < hud_cut);
    return float4(rejected ? float3(1.0, 0.85, 0.0) : base, fade);
}
)__";

static bool BuildDebug(device *dev, device_state &ds, format rt_format)
{
    if (ds.dbg_pso != 0 && ds.dbg_rt_format == rt_format)
        return true;
    if (ds.dbg_tried && ds.dbg_rt_format == rt_format)
        return false;
    ds.dbg_tried = true;

    if (ds.dbg_pso != 0)    { dev->destroy_pipeline(ds.dbg_pso); ds.dbg_pso = { 0 }; }
    if (ds.dbg_layout != 0) { dev->destroy_pipeline_layout(ds.dbg_layout); ds.dbg_layout = { 0 }; }

    const device_api api = dev->get_api();
    if (api != device_api::d3d10 && api != device_api::d3d11 && api != device_api::d3d12)
        return false;

    ID3DBlob *vs = nullptr, *ps = nullptr, *err = nullptr;
    if (FAILED(D3DCompile(kDebugHLSL, strlen(kDebugHLSL), nullptr, nullptr, nullptr,
                          "vs_main", "vs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &vs, &err)) ||
        FAILED(D3DCompile(kDebugHLSL, strlen(kDebugHLSL), nullptr, nullptr, nullptr,
                          "ps_main", "ps_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &ps, &err)))
    {
        Log("[depth-charge] could not compile the inspector shader: %s",
            err ? (const char *)err->GetBufferPointer() : "(no message)");
        if (vs) vs->Release(); if (ps) ps->Release(); if (err) err->Release();
        return false;
    }
    if (err) { err->Release(); err = nullptr; }

    descriptor_range srv_range = {};
    srv_range.binding = 0; srv_range.dx_register_index = 0; srv_range.count = 2;
    srv_range.type = descriptor_type::shader_resource_view;
    srv_range.visibility = shader_stage::pixel;

    descriptor_range smp_range = {};
    smp_range.binding = 0; smp_range.dx_register_index = 0; smp_range.count = 1;
    smp_range.type = descriptor_type::sampler;
    smp_range.visibility = shader_stage::pixel;

    constant_range cb_range = {};
    cb_range.binding = 0; cb_range.dx_register_index = 0; cb_range.count = 16;   // (Oct 23 2026: + enc)
    cb_range.visibility = shader_stage::pixel;

    const pipeline_layout_param params[3] = {
        pipeline_layout_param(srv_range),
        pipeline_layout_param(smp_range),
        pipeline_layout_param(cb_range),
    };
    if (!dev->create_pipeline_layout(3, params, &ds.dbg_layout))
    { vs->Release(); ps->Release(); return false; }

    shader_desc vs_desc = {}; vs_desc.code = vs->GetBufferPointer(); vs_desc.code_size = vs->GetBufferSize();
    shader_desc ps_desc = {}; ps_desc.code = ps->GetBufferPointer(); ps_desc.code_size = ps->GetBufferSize();

    blend_desc         blend = {};
    blend.blend_enable[0]      = true;                 // so the reveal can fade in
    blend.source_color_blend_factor[0] = blend_factor::source_alpha;
    blend.dest_color_blend_factor[0]   = blend_factor::one_minus_source_alpha;
    blend.color_blend_op[0]            = blend_op::add;
    blend.source_alpha_blend_factor[0] = blend_factor::one;
    blend.dest_alpha_blend_factor[0]   = blend_factor::zero;
    blend.alpha_blend_op[0]            = blend_op::add;
    blend.render_target_write_mask[0] = 0x0F;
    rasterizer_desc    raster = {};  raster.cull_mode = cull_mode::none;
    depth_stencil_desc depth = {};   depth.depth_enable = false; depth.stencil_enable = false;
    const primitive_topology topology = primitive_topology::triangle_list;
    const uint32_t sample_mask = 0xFFFFFFFF;

    const pipeline_subobject subobjects[] = {
        { pipeline_subobject_type::vertex_shader,         1, (void *)&vs_desc   },
        { pipeline_subobject_type::pixel_shader,          1, (void *)&ps_desc   },
        { pipeline_subobject_type::render_target_formats, 1, (void *)&rt_format },
        { pipeline_subobject_type::blend_state,           1, (void *)&blend     },
        { pipeline_subobject_type::rasterizer_state,      1, (void *)&raster    },
        { pipeline_subobject_type::depth_stencil_state,   1, (void *)&depth     },
        { pipeline_subobject_type::primitive_topology,    1, (void *)&topology  },
        { pipeline_subobject_type::sample_mask,           1, (void *)&sample_mask },
    };

    const bool ok = dev->create_pipeline(ds.dbg_layout,
                                         (uint32_t)(sizeof(subobjects) / sizeof(subobjects[0])),
                                         subobjects, &ds.dbg_pso);
    vs->Release(); ps->Release();
    if (!ok)
    {
        dev->destroy_pipeline_layout(ds.dbg_layout); ds.dbg_layout = { 0 };
        Log("[depth-charge] could not create the inspector pipeline");
        return false;
    }

    ds.dbg_rt_format = rt_format;
    ds.dbg_tried = false;
    Log("[depth-charge] inspector ready");
    return true;
}

static void on_finish_effects(effect_runtime *runtime, command_list *cmd_list,
                              resource_view rtv, resource_view)
{
    const int view = (g_cal_state == CAL_SHOWING) ? 2 : g_dbg_view;   // 2 = false colour
    if (view <= 0 || cmd_list == nullptr || rtv == 0)
        return;

    device *const dev = runtime->get_device();
    std::lock_guard<std::mutex> lock(g_mutex);
    device_state &ds = g_devices[dev];

    const resource_view raw_srv = (ds.disp_srv != 0) ? ds.disp_srv : ds.lin_srv;
    const resource_view lin_srv = (ds.lin_srv  != 0) ? ds.lin_srv  : ds.disp_srv;
    if (raw_srv == 0 || lin_srv == 0 || ds.convert_sampler == 0)
        return;

    const resource rt = dev->get_resource_from_view(rtv);
    if (rt == 0) return;
    const resource_desc rtd = dev->get_resource_desc(rt);
    if (!BuildDebug(dev, ds, format_to_default_typed(rtd.texture.format, 0)))
        return;

    cmd_list->bind_render_targets_and_depth_stencil(1, &rtv);

    const viewport vp = { 0.0f, 0.0f, (float)rtd.texture.width, (float)rtd.texture.height, 0.0f, 1.0f };
    cmd_list->bind_viewports(0, 1, &vp);
    const rect sc = { 0, 0, (int32_t)rtd.texture.width, (int32_t)rtd.texture.height };
    cmd_list->bind_scissor_rects(0, 1, &sc);
    cmd_list->bind_pipeline(pipeline_stage::all_graphics, ds.dbg_pso);

    const resource_view srvs[2] = { raw_srv, lin_srv };
    descriptor_table_update u = {};
    u.binding = 0; u.count = 2; u.type = descriptor_type::shader_resource_view; u.descriptors = srvs;
    cmd_list->push_descriptors(shader_stage::pixel, ds.dbg_layout, 0, u);

    descriptor_table_update su = {};
    su.binding = 0; su.count = 1; su.type = descriptor_type::sampler; su.descriptors = &ds.convert_sampler;
    cmd_list->push_descriptors(shader_stage::pixel, ds.dbg_layout, 1, su);

    const float aspect = (rtd.texture.height > 0)
        ? (float)rtd.texture.width / (float)rtd.texture.height : 1.7778f;
    // A debug view replacing the frame in one step is genuinely unpleasant to look at.
    // Manual views come up solid; the automatic reveal ramps in and back out.
    float fade = 1.0f;
    if (g_cal_state == CAL_SHOWING)
    {
        const float t = (float)(((double)GetTickCount64() / 1000.0) - g_cal_started);
        fade = (t < 0.5f) ? (t / 0.5f) : (t > 2.5f ? (3.0f - t) / 0.5f : 1.0f);
        fade = fade < 0.0f ? 0.0f : (fade > 1.0f ? 1.0f : fade);
    }

    const float constants[16] = {
        (float)view, g_dbg_log_min, g_dbg_log_max, g_dbg_near_gain,
        g_dbg_bands, g_dbg_hud_cut, tanf(g_dbg_fov * 0.5f * 3.14159265f / 180.0f), aspect,
        fade, g_enc[4], g_enc[5], 0.0f,
        g_enc[0], g_enc[1], g_enc[2], g_enc[3],
    };
    cmd_list->push_constants(shader_stage::pixel, ds.dbg_layout, 2, 0, 16, constants);

    cmd_list->draw(3, 1, 0, 0);
}

// stage 2: place it where the picture actually is

// Hand the derived cutoff to Lumenite. The uniform handle is only valid until effects
// reload, so this is never cached - it is a name lookup on the frames it runs, and it
// stops running as soon as it lands. If the shader is not loaded (or not present at
// all) the lookup fails harmlessly and the budget runs out.
// Whether a loaded effect has a cutoff to receive: -1 not known yet, 0 searched the whole
// budget and found none (stock LumeniteFX has no such setting), 1 found. Only the panel reads it.
static int g_cutoff_target = -1;

static void PushOverlayCutoff(effect_runtime *runtime)
{
    if (g_cutoff_tries <= 0 || g_cfg_overlay_cutoff < 0.0f || runtime == nullptr)
        return;
    --g_cutoff_tries;
    const effect_uniform_variable var =
        runtime->find_uniform_variable("lumenite_Kernel.fx", "OVERLAY_NEAR_CUTOFF");
    if (var == 0)
    {
        if (g_cutoff_tries == 0)
            g_cutoff_target = 0;
        return;
    }
    runtime->set_uniform_value_float(var, &g_cfg_overlay_cutoff, 1);
    g_cutoff_target = 1;
    g_cutoff_tries = 0;
    Log("[depth-charge] pushed overlay cutoff %.5f into lumenite_Kernel.fx", g_cfg_overlay_cutoff);
}

// Every Lumenite effect includes lumenite_Projections.fxh, so each one carries its own
// copy of the uniform and each one needs telling. Getting only some of them is worse
// than getting none: the Kernel builds normals in one frustum while SSSR marches rays
// through another, and position and normal end up in different spaces.
static const char *kLumeniteEffects[] = {
    "lumenite_Kernel.fx", "lumenite_LSAO.fx", "lumenite_SSSR.fx",
    "lumenite_RTAO.fx",   "lumenite_QuantAO.fx", "lumenite_PocketAO.fx",
};

static void PushPictureAspect(effect_runtime *runtime)
{
    if (g_aspect_tries <= 0 || g_picture_aspect <= 0.0f || runtime == nullptr)
        return;
    --g_aspect_tries;

    int found = 0;
    for (const char *fx : kLumeniteEffects)
    {
        const effect_uniform_variable var =
            runtime->find_uniform_variable(fx, "PICTURE_ASPECT_OVERRIDE");
        if (var == 0) continue;
        runtime->set_uniform_value_float(var, &g_picture_aspect, 1);
        ++found;
    }
    if (found == 0)
        return;                     // effects not compiled yet; try again next frame

    g_aspect_tries = 0;
    if (g_pushed_aspect != g_picture_aspect)
    {
        g_pushed_aspect = g_picture_aspect;
        Log("[depth-charge] pushed picture aspect %.4f into %d Lumenite effect(s)",
            g_picture_aspect, found);
    }
}

// (Oct 23 2026) ReShade's own depth settings, so Linearise can write depth the way they read it.
static void ReadDepthSettings(effect_runtime *runtime)
{
    if (runtime == nullptr)
        return;
    struct Def { const char *name; float fallback; };
    static const Def kDefs[6] = {
        { "RESHADE_DEPTH_LINEARIZATION_FAR_PLANE", 1000.0f },
        { "RESHADE_DEPTH_INPUT_IS_REVERSED",       1.0f },
        { "RESHADE_DEPTH_MULTIPLIER",              1.0f },
        { "RESHADE_DEPTH_INPUT_IS_LOGARITHMIC",    0.0f },
        { "RESHADE_DEPTH_INPUT_IS_UPSIDE_DOWN",    0.0f },
        { "RESHADE_DEPTH_INPUT_IS_MIRRORED",       0.0f },
    };
    float v[6];
    for (int i = 0; i < 6; ++i)
    {
        char s[64] = "";
        float x = kDefs[i].fallback, t = 0.0f;
        if (runtime->get_preprocessor_definition(kDefs[i].name, s) && sscanf_s(s, "%f", &t) == 1)
            x = t;
        v[i] = x;
    }
    if (!(v[0] > 0.0f)) v[0] = 1000.0f;            // ReShade divides by it: nothing sensible at or below zero
    if (!(v[2] > 0.0f)) v[2] = 1.0f;               // and the way back divides by the multiplier
    for (int i = 1; i < 6; ++i)
        if (i != 2) v[i] = (v[i] != 0.0f) ? 1.0f : 0.0f;

    bool changed = false;
    for (int i = 0; i < 6; ++i)
        if (g_enc[i] != v[i]) { g_enc[i] = v[i]; changed = true; }
    static bool s_logged = false;
    if (!changed && s_logged)
        return;
    s_logged = true;
    if (g_enc[0] == 1.0f && g_enc[1] == 0.0f && g_enc[2] == 1.0f && g_enc[3] == 0.0f && g_enc[4] == 0.0f && g_enc[5] == 0.0f)
        Log("[depth-charge] ReShade's depth settings are neutral (far plane 1, not reversed, multiplier 1): "
            "publishing the distance as it is");
    else
        Log("[depth-charge] ReShade's depth settings: far plane %g, %s, multiplier %g%s%s%s. Publishing depth "
            "written for them, so effects read it back as Depth Charge's distance",
            g_enc[0], g_enc[1] != 0.0f ? "reversed" : "not reversed", g_enc[2],
            g_enc[3] != 0.0f ? ", logarithmic" : "", g_enc[4] != 0.0f ? ", upside down" : "",
            g_enc[5] != 0.0f ? ", mirrored" : "");

    // Scales and offsets move where ReShade reads. Depth Charge does not undo those, so say so.
    static const char *const kMoves[6] = {
        "RESHADE_DEPTH_INPUT_X_SCALE", "RESHADE_DEPTH_INPUT_Y_SCALE", "RESHADE_DEPTH_INPUT_X_OFFSET",
        "RESHADE_DEPTH_INPUT_Y_OFFSET", "RESHADE_DEPTH_INPUT_X_PIXEL_OFFSET", "RESHADE_DEPTH_INPUT_Y_PIXEL_OFFSET" };
    for (int i = 0; i < 6; ++i)
    {
        char s[64] = "";
        float t = 0.0f;
        if (!runtime->get_preprocessor_definition(kMoves[i], s) || sscanf_s(s, "%f", &t) != 1)
            continue;
        const float want = (i < 2) ? 1.0f : 0.0f;
        if (t != want)
            Log("[depth-charge] %s=%s moves where ReShade reads depth. Depth Charge does not undo that, "
                "so set it back to %g", kMoves[i], s, want);
    }
}

// A recompile resets uniforms to their preset values, so ask to be re-applied.
static void on_reloaded_effects(effect_runtime *runtime)
{
    ReadDepthSettings(runtime);
    g_cutoff_tries = 1200;
    g_cutoff_target = -1;
    g_aspect_tries = 1200;
    g_pushed_aspect = -1.0f;
}

// g_devices keeps every device the game ever made, and BioShock Remastered probes
// several before settling. Reading "the first one" from the map is therefore reading an
// arbitrary, usually dead, device. The live one is the one presenting frames.
static device_state *LiveDevice()          // call with g_mutex held
{
    device_state *best = nullptr;
    for (auto &kv : g_devices)
        if (best == nullptr || kv.second.frames > best->frames)
            best = &kv.second;
    return best;
}

// Auto picture aspect from the black bars (Oct 21 2026).
//
// "Auto" used to read the viewport of the last draw into the window. PCSX2 2.8 draws its picture
// with a full-window viewport and places it with the corners of the quad it draws, so that viewport
// says "the whole window" and depth was stretched across the bars (measured 21 Oct: every back-buffer
// draw in Kingdom Hearts was 5120x2160 at 0,0 on a 5120x2160 window).
//
// Emulators paint the bars black, so Auto looks for them in the finished frame. A scan copies the
// back buffer, a shader boils it down to 1024 column bins and 1024 row bins, each holding the
// fraction of its samples that are not black, and the add-on reads that back a second later, when
// the GPU has long finished it, so nothing waits on the GPU (MGS3, 21 Oct: 0.09 ms of CPU per scan).
//
// It only looks when something changed: Auto switched on, Check again pressed in the panel, the game
// changed, or the window changed size. It then scans once a second until three scans in a row agree
// on a shape, or ten scans gave no agreement, or two minutes passed, and stops until the next change.
// Changing an emulator's own aspect setting changes none of those, so the panel has Check again for
// that. (Picking Auto in the list when it is already Auto does nothing: the list only reports a
// change, measured 21 Oct.) The answer is kept as a shape, not a rectangle, so it survives a resize.
//
// What a scan counts, from the bar survey on MGS3 (21 Oct):
// - The picture is the columns and rows where at least 25% of samples are not black. That ignores
//   the emulator's own messages drawn over the bars (PCSX2's save-state and FPS text sat at x=15).
// - An all-black frame (a fade, a loading screen) does not count.
// - The answer is one of the panel's shapes (4:3, 16:10, 16:9, the whole window): the smallest one
//   that holds everything seen and that the picture fills between its bars. A game's own letterbox
//   (MGS3's cutscenes, 3840x1545) still fills 16:9 from side to side, so it still reads 16:9. A logo
//   in the middle of a black screen fills nothing, so it does not count.
// - The emulator's pause menu covers the bars too, so it reads as the whole window. Left alone: it
//   only matters if a change happens while that menu is up, and the shape can be picked by hand.
// auto_bars=0 in depth-charge.cfg turns this off, and Auto goes back to the viewport.
static std::atomic<bool> g_auto_repick{ false };    // the panel sets this when Auto is picked or Check again is pressed
static float g_auto_found = 0.0f;                   // 0 = nothing yet, -1 = the whole window, else width / height
static char  g_auto_status[128] = "Auto: waiting for the game's picture";

static const int kBarBins = 1024;
static const char *kBarHLSL = R"__(
Texture2D<float4> src : register(t0);

void vs_main(uint id : SV_VertexID, out float4 pos : SV_Position)
{
    float2 uv = float2((id == 2) ? 2.0 : 0.0, (id == 1) ? 2.0 : 0.0);
    pos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
}

// Row 0 of the target: one texel per column bin. Row 1: one per row bin.
// x = fraction of samples that are not black, y = the brightest sample.
float2 ps_main(float4 pos : SV_Position) : SV_Target
{
    uint w, h;
    src.GetDimensions(w, h);
    const uint bin    = (uint)pos.x;
    const bool cols   = pos.y < 1.0;
    const uint along  = cols ? h : w;              // swept by 64 samples
    const uint across = cols ? w : h;              // split into the 1024 bins
    const uint b0 = bin * across / 1024;
    const uint b1 = max(b0 + 1, (bin + 1) * across / 1024);
    uint hit = 0, total = 0;
    float brightest = 0.0;
    [loop] for (uint i = 0; i < 64; i++)
    {
        const uint a = ((2 * i + 1) * along) / 128;
        [loop] for (uint j = 0; j < 4; j++)
        {
            const uint c = b0 + (j * (b1 - b0)) / 4;
            const uint2 p = cols ? uint2(c, a) : uint2(a, c);
            const float3 rgb = src.Load(int3(p, 0)).rgb;
            const float v = max(rgb.r, max(rgb.g, rgb.b));
            brightest = max(brightest, v);
            hit += (v > 2.0 / 255.0) ? 1 : 0;
            total++;
        }
    }
    return float2((float)hit / (float)total, brightest);
}
)__";

static bool BuildBarScan(device *dev, device_state &ds)
{
    if (ds.bar_pso != 0) return true;
    if (ds.bar_tried)    return false;
    ds.bar_tried = true;

    const device_api api = dev->get_api();
    if (api != device_api::d3d10 && api != device_api::d3d11 && api != device_api::d3d12)
    {
        Log("[depth-charge] Auto can't look for black bars without Direct3D 10/11/12, so it uses the viewport");
        return false;
    }

    ID3DBlob *vs = nullptr, *ps = nullptr, *err = nullptr;
    if (FAILED(D3DCompile(kBarHLSL, strlen(kBarHLSL), nullptr, nullptr, nullptr,
                          "vs_main", "vs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &vs, &err)) ||
        FAILED(D3DCompile(kBarHLSL, strlen(kBarHLSL), nullptr, nullptr, nullptr,
                          "ps_main", "ps_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &ps, &err)))
    {
        Log("[depth-charge] Auto could not compile the black-bar shader: %s",
            err ? (const char *)err->GetBufferPointer() : "(no message)");
        if (vs) vs->Release();
        if (ps) ps->Release();
        if (err) err->Release();
        return false;
    }
    if (err) { err->Release(); err = nullptr; }

    descriptor_range srv = {};
    srv.binding = 0; srv.dx_register_index = 0; srv.count = 1;
    srv.type = descriptor_type::shader_resource_view; srv.visibility = shader_stage::pixel;
    const pipeline_layout_param params[1] = { pipeline_layout_param(srv) };
    if (!dev->create_pipeline_layout(1, params, &ds.bar_layout))
    {
        vs->Release(); ps->Release();
        return false;
    }

    shader_desc vsd = {}; vsd.code = vs->GetBufferPointer(); vsd.code_size = vs->GetBufferSize();
    shader_desc psd = {}; psd.code = ps->GetBufferPointer(); psd.code_size = ps->GetBufferSize();
    const format rtf = format::r32g32_float;
    blend_desc bl = {}; rasterizer_desc rs = {}; rs.cull_mode = cull_mode::none;
    depth_stencil_desc dss = {}; dss.depth_enable = false; dss.stencil_enable = false;
    const primitive_topology topo = primitive_topology::triangle_list;
    const uint32_t mask = 0xFFFFFFFF;
    const pipeline_subobject so[] = {
        { pipeline_subobject_type::vertex_shader,         1, (void *)&vsd },
        { pipeline_subobject_type::pixel_shader,          1, (void *)&psd },
        { pipeline_subobject_type::render_target_formats, 1, (void *)&rtf },
        { pipeline_subobject_type::blend_state,           1, (void *)&bl  },
        { pipeline_subobject_type::rasterizer_state,      1, (void *)&rs  },
        { pipeline_subobject_type::depth_stencil_state,   1, (void *)&dss },
        { pipeline_subobject_type::primitive_topology,    1, (void *)&topo },
        { pipeline_subobject_type::sample_mask,           1, (void *)&mask },
    };
    const bool ok = dev->create_pipeline(ds.bar_layout, 8, so, &ds.bar_pso);
    vs->Release(); ps->Release();
    if (!ok) { dev->destroy_pipeline_layout(ds.bar_layout); ds.bar_layout = { 0 }; return false; }

    resource_desc rd(kBarBins, 2, 1, 1, format::r32g32_float, 1, memory_heap::default_,
                     resource_usage::render_target | resource_usage::copy_source);
    if (!dev->create_resource(rd, nullptr, resource_usage::copy_source, &ds.bar_rt) ||
        !dev->create_resource_view(ds.bar_rt, resource_usage::render_target,
                                   resource_view_desc(format::r32g32_float), &ds.bar_rtv))
    { Log("[depth-charge] Auto could not create the black-bar target"); return false; }

    resource_desc sd(kBarBins, 2, 1, 1, format::r32g32_float, 1, memory_heap::readback,
                     resource_usage::copy_dest);
    if (!dev->create_resource(sd, nullptr, resource_usage::copy_dest, &ds.bar_stage))
    { Log("[depth-charge] Auto could not create the black-bar readback buffer"); return false; }

    Log("[depth-charge] Auto aspect: black-bar detection ready");
    return true;
}

// A private copy of the back buffer the shader can read: back buffers usually cannot be sampled.
static bool BarCopyFor(device *dev, device_state &ds, const resource_desc &bd)
{
    const format typeless = format_to_typeless(bd.texture.format);
    if (ds.bar_copy != 0 && ds.bar_copy_w == bd.texture.width && ds.bar_copy_h == bd.texture.height &&
        ds.bar_copy_fmt == typeless)
        return true;
    if (ds.bar_copy_srv != 0) { dev->destroy_resource_view(ds.bar_copy_srv); ds.bar_copy_srv = { 0 }; }
    if (ds.bar_copy != 0)     { dev->destroy_resource(ds.bar_copy); ds.bar_copy = { 0 }; }
    ds.bar_copy_w = 0; ds.bar_copy_h = 0; ds.bar_copy_fmt = format::unknown;

    resource_desc cd(bd.texture.width, bd.texture.height, 1, 1, typeless, 1, memory_heap::default_,
                     resource_usage::copy_dest | resource_usage::shader_resource);
    if (!dev->create_resource(cd, nullptr, resource_usage::shader_resource, &ds.bar_copy))
    {
        Log("[depth-charge] Auto could not create a %ux%u copy of the frame (format %u)",
            bd.texture.width, bd.texture.height, (unsigned)bd.texture.format);
        return false;
    }
    if (!dev->create_resource_view(ds.bar_copy, resource_usage::shader_resource,
                                   resource_view_desc(format_to_default_typed(typeless, 0)), &ds.bar_copy_srv))
    {
        dev->destroy_resource(ds.bar_copy); ds.bar_copy = { 0 };
        Log("[depth-charge] Auto could not view the copy of the frame (format %u)", (unsigned)bd.texture.format);
        return false;
    }
    ds.bar_copy_w = bd.texture.width; ds.bar_copy_h = bd.texture.height; ds.bar_copy_fmt = typeless;
    return true;
}

// First and last bin whose non-black fraction reaches 'need' (-1, -1 if none).
static void BarExtent(const float *frac, float need, int &lo, int &hi)
{
    lo = -1; hi = -1;
    for (int i = 0; i < kBarBins; ++i)
        if (frac[i] >= need) { if (lo < 0) lo = i; hi = i; }
}

static void BarRectText(char *s, size_t n, int cl, int ch, int rl, int rh, uint32_t w, uint32_t h)
{
    if (cl < 0 || rl < 0) { sprintf_s(s, n, "none (frame is all black)"); return; }
    const uint32_t x0 = (uint32_t)((uint64_t)cl * w / kBarBins), x1 = (uint32_t)((uint64_t)(ch + 1) * w / kBarBins);
    const uint32_t y0 = (uint32_t)((uint64_t)rl * h / kBarBins), y1 = (uint32_t)((uint64_t)(rh + 1) * h / kBarBins);
    sprintf_s(s, n, "%ux%u at %u,%u (aspect %.3f)", x1 - x0, y1 - y0, x0, y0,
              (y1 > y0) ? (double)(x1 - x0) / (double)(y1 - y0) : 0.0);
}

// The panel's shapes, as Auto names them. -1 is the whole window (the panel's "Full frame").
struct AutoShape { float aspect; const char *name; };
static const AutoShape kAutoShapes[] = {
    { 4.0f / 3.0f,   "4:3" },
    { 16.0f / 10.0f, "16:10" },
    { 16.0f / 9.0f,  "16:9" },
    { -1.0f,         "the whole window" },
};

static const char *AutoShapeName(float a)
{
    for (const AutoShape &s : kAutoShapes)
        if (fabsf(s.aspect - a) < 0.0005f) return s.name;
    return "an unknown shape";
}

// Where an emulator puts a picture of this shape in a W x H window: fitted and centred.
static void AutoFit(float a, uint32_t W, uint32_t H, uint32_t &x, uint32_t &y, uint32_t &w, uint32_t &h)
{
    x = 0; y = 0; w = W; h = H;
    if (a <= 0.0f || W == 0 || H == 0) return;
    if (a < (float)W / (float)H) { w = (uint32_t)(H * a + 0.5f); if (w > W) w = W; }
    else                         { h = (uint32_t)(W / a + 0.5f); if (h > H) h = H; }
    x = (W - w) / 2; y = (H - h) / 2;
}

// Grade one scan: the shape it shows, or 0 when it does not count. 'seen' gets the picture found.
static float AutoJudge(const float *colf, const float *rowf, uint32_t W, uint32_t H, char *seen, size_t seen_n)
{
    int cl, ch, rl, rh;
    BarExtent(colf, 0.25f, cl, ch);
    BarExtent(rowf, 0.25f, rl, rh);
    BarRectText(seen, seen_n, cl, ch, rl, rh, W, H);
    if (cl < 0 || rl < 0 || W == 0 || H == 0)
        return 0.0f;
    const uint32_t x0 = (uint32_t)((uint64_t)cl * W / kBarBins), x1 = (uint32_t)((uint64_t)(ch + 1) * W / kBarBins);
    const uint32_t y0 = (uint32_t)((uint64_t)rl * H / kBarBins), y1 = (uint32_t)((uint64_t)(rh + 1) * H / kBarBins);
    const uint32_t sx = 4 * W / kBarBins + 1, sy = 4 * H / kBarBins + 1;     // four bins of slack for rounding

    // Which directions can tell the shapes apart in this window: across when some shape would have
    // bars at the sides, up and down when some shape would have bars above and below.
    bool cols_tell = false, rows_tell = false;
    for (const AutoShape &s : kAutoShapes)
    {
        uint32_t x, y, w, h;
        AutoFit(s.aspect, W, H, x, y, w, h);
        if (W - w > 2 * sx) cols_tell = true;
        if (H - h > 2 * sy) rows_tell = true;
    }

    float best = 0.0f;
    uint64_t best_area = 0;
    for (const AutoShape &s : kAutoShapes)
    {
        uint32_t x, y, w, h;
        AutoFit(s.aspect, W, H, x, y, w, h);
        // Everything seen must sit inside this shape: there is no picture in a black bar.
        if (x0 + sx < x || x1 > x + w + sx || y0 + sy < y || y1 > y + h + sy)
            continue;
        // The picture must reach from bar to bar: across where this shape has bars at the sides, up
        // and down where it has them above and below. A shape with no bars in this window must be
        // filled in every direction that tells the shapes apart.
        const bool side_bars = W - w > 2 * sx, top_bars = H - h > 2 * sy;
        const bool fills_x = (uint64_t)(x1 - x0) * 100 >= (uint64_t)w * 97;
        const bool fills_y = (uint64_t)(y1 - y0) * 100 >= (uint64_t)h * 97;
        bool ok;
        if (side_bars || top_bars)
            ok = (!side_bars || fills_x) && (!top_bars || fills_y);
        else
            ok = (cols_tell || rows_tell) && (!cols_tell || fills_x) && (!rows_tell || fills_y);
        if (!ok)
            continue;
        const uint64_t area = (uint64_t)w * h;
        if (best == 0.0f || area < best_area) { best = s.aspect; best_area = area; }
    }
    return best;
}

static void AutoAspectTick(device *dev, effect_runtime *runtime, device_state &ds, uint32_t sw, uint32_t sh)
{
    static bool        s_was_auto = false, s_busy = false;
    static std::string s_game;
    static uint32_t    s_w = 0, s_h = 0;
    static ULONGLONG   s_next = 0, s_give_up = 0;
    static int         s_scans = 0, s_counted = 0, s_streak = 0;
    static float       s_candidate = 0.0f;
    static double      s_ms_max = 0.0;

    if (!g_cfg_auto_bars || runtime == nullptr)
        return;
    const bool repick = g_auto_repick.exchange(false);
    if (g_cfg_aspect != 0.0f)
    {
        s_was_auto = false;
        s_busy = false;
        return;
    }

    // 1. Something changed: start looking.
    const char *why = nullptr;
    if (repick)                       why = "asked from the panel";
    else if (!s_was_auto)             why = "Auto is on";
    else if (g_game_key != s_game)    why = "the game changed";
    else if (sw != s_w || sh != s_h)  why = "the window changed size";
    s_was_auto = true; s_game = g_game_key; s_w = sw; s_h = sh;
    const ULONGLONG now = GetTickCount64();
    if (why != nullptr)
    {
        s_busy = true;
        s_next = 0;
        s_give_up = now + 120000;
        s_scans = 0; s_counted = 0; s_streak = 0; s_candidate = 0.0f; s_ms_max = 0.0;
        ds.bar_job_pending = false;           // a scan from before the change says nothing about after it
        sprintf_s(g_auto_status, "Auto: looking for the picture...");
        Log("[depth-charge] Auto: looking for the black bars around the picture (%s)", why);
    }
    if (!s_busy || now < s_next)
        return;
    s_next = now + 1000;
    if (!BuildBarScan(dev, ds))
    {
        s_busy = false;
        sprintf_s(g_auto_status, "Auto: can't look for black bars here (needs Direct3D 10, 11 or 12)");
        return;
    }

    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);

    // 2. Read the scan issued a second ago. The GPU finished it long since, so nothing waits here.
    float found = 0.0f;
    bool  have = false;
    char  seen[96] = "";
    if (ds.bar_job_pending)
    {
        ds.bar_job_pending = false;
        subresource_data map = {};
        if (dev->map_texture_region(ds.bar_stage, 0, nullptr, map_access::read_only, &map))
        {
            std::vector<float> colf(kBarBins), rowf(kBarBins);
            const float *r0 = (const float *)map.data;
            const float *r1 = (const float *)((const uint8_t *)map.data + map.row_pitch);
            for (int i = 0; i < kBarBins; ++i) { colf[i] = r0[i * 2]; rowf[i] = r1[i * 2]; }
            dev->unmap_texture_region(ds.bar_stage, 0);
            found = AutoJudge(colf.data(), rowf.data(), ds.bar_job_w, ds.bar_job_h, seen, sizeof(seen));
            have = true;
        }
    }

    // 3. Issue this second's scan. It goes to the GPU with the frame and is read on the next tick.
    const resource bb = runtime->get_current_back_buffer();
    command_list *const cmd = runtime->get_command_queue()->get_immediate_command_list();
    if (bb != 0 && cmd != nullptr)
    {
        const resource_desc bd = dev->get_resource_desc(bb);
        if (bd.type == resource_type::texture_2d && bd.texture.samples <= 1 &&
            bd.texture.width >= 64 && bd.texture.height >= 64 && BarCopyFor(dev, ds, bd))
        {
            // The back buffer is in render_target state around effects: ReShade copies it from that state too.
            {
                const resource res[2] = { bb, ds.bar_copy };
                const resource_usage from[2] = { resource_usage::render_target, resource_usage::shader_resource };
                const resource_usage to[2]   = { resource_usage::copy_source,   resource_usage::copy_dest };
                cmd->barrier(2, res, from, to);
                cmd->copy_texture_region(bb, 0, nullptr, ds.bar_copy, 0, nullptr);
                cmd->barrier(2, res, to, from);
            }
            const resource rt = ds.bar_rt;
            const resource_usage as_copy = resource_usage::copy_source, as_target = resource_usage::render_target;
            cmd->barrier(1, &rt, &as_copy, &as_target);
            cmd->bind_render_targets_and_depth_stencil(1, &ds.bar_rtv);
            const viewport vp = { 0.0f, 0.0f, (float)kBarBins, 2.0f, 0.0f, 1.0f };
            cmd->bind_viewports(0, 1, &vp);
            const rect sc = { 0, 0, kBarBins, 2 };
            cmd->bind_scissor_rects(0, 1, &sc);
            cmd->bind_pipeline(pipeline_stage::all_graphics, ds.bar_pso);
            descriptor_table_update u = {};
            u.binding = 0; u.count = 1; u.type = descriptor_type::shader_resource_view; u.descriptors = &ds.bar_copy_srv;
            cmd->push_descriptors(shader_stage::pixel, ds.bar_layout, 0, u);
            cmd->draw(3, 1, 0, 0);
            cmd->barrier(1, &rt, &as_target, &as_copy);
            cmd->copy_texture_region(ds.bar_rt, 0, nullptr, ds.bar_stage, 0, nullptr);
            ds.bar_job_w = bd.texture.width; ds.bar_job_h = bd.texture.height;
            ds.bar_job_pending = true;
        }
    }

    QueryPerformanceCounter(&t1);
    const double ms = 1000.0 * (double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart;
    if (ms > s_ms_max)
        s_ms_max = ms;

    // 4. Count it. Three of the same shape in a row settles it.
    if (have)
    {
        ++s_scans;
        if (found == 0.0f)
        {
            Log("[depth-charge] Auto scan %d: %s, not counted", s_scans,
                strncmp(seen, "none", 4) == 0 ? "the frame is all black" : seen);
        }
        else
        {
            ++s_counted;
            if (found == s_candidate) ++s_streak;
            else { s_candidate = found; s_streak = 1; }
            Log("[depth-charge] Auto scan %d: picture %s, which is %s (%d in a row)",
                s_scans, seen, AutoShapeName(found), s_streak);
            if (s_streak >= 3)
            {
                s_busy = false;
                if (g_auto_found != found)
                {
                    g_auto_found = found;
                    ds.logged_rect = -1;
                }
                sprintf_s(g_auto_status, "Auto: %s (found from the black bars)", AutoShapeName(found));
                Log("[depth-charge] Auto: the picture is %s (3 scans agreed, %d scans in all, slowest %.3f ms)",
                    AutoShapeName(found), s_scans, s_ms_max);
                return;
            }
        }
    }
    if (s_counted >= 10 || now >= s_give_up)
    {
        s_busy = false;
        if (g_auto_found != 0.0f)
            sprintf_s(g_auto_status, "Auto: no clear answer this time, keeping %s", AutoShapeName(g_auto_found));
        else
            sprintf_s(g_auto_status, "Auto: no clear answer. If effects look shifted, pick the shape by hand.");
        Log("[depth-charge] Auto: no clear answer after %d scans (%d counted), keeping %s", s_scans, s_counted,
            g_auto_found != 0.0f ? AutoShapeName(g_auto_found) : "the viewport");
    }
}

static void on_begin_effects(effect_runtime *runtime, command_list *cmd_list, resource_view, resource_view)
{
    // ReShade's depth settings: re-read once a second (a string lookup) as well as after every reload,
    // so the depth published this frame is always written for the settings the effects use.
    {
        static ULONGLONG s_next = 0;
        const ULONGLONG now = GetTickCount64();
        if (now >= s_next) { s_next = now + 1000; ReadDepthSettings(runtime); }
    }

    PushOverlayCutoff(runtime);
    PushPictureAspect(runtime);

    device *const dev = runtime->get_device();
    std::lock_guard<std::mutex> lock(g_mutex);
    device_state &ds = g_devices[dev];
    ds.runtime = runtime;

    if (!ds.have_raw || ds.raw == 0)
        return;

    const uint32_t sw = ds.bb_w ? ds.bb_w : ds.raw_w;
    const uint32_t sh = ds.bb_h ? ds.bb_h : ds.raw_h;

    // The rectangle of the window the emulator actually draws its picture into.
    uint32_t rx = 0, ry = 0, rw = sw, rh = sh;
    float aspect = g_cfg_aspect;
    if (aspect == -2.0f && ds.raw_w && ds.raw_h)
        aspect = (float)ds.raw_w / (float)ds.raw_h;
    if (aspect == 0.0f && g_auto_found != 0.0f)
        aspect = g_auto_found;      // Auto: the shape the black bars showed (-1 = the whole window)
    if (aspect > 0.0f)
    {
        // Told explicitly what shape the picture is: fit it, centred.
        const float screen_aspect = (float)sw / (float)sh;
        if (aspect < screen_aspect) { rh = sh; rw = (uint32_t)(sh * aspect + 0.5f); }
        else                              { rw = sw; rh = (uint32_t)(sw / aspect + 0.5f); }
        if (rw > sw) rw = sw;
        if (rh > sh) rh = sh;
        rx = (sw - rw) / 2; ry = (sh - rh) / 2;
    }
    else if (aspect == 0.0f && ds.display_rect_set &&
        ds.display_rect.width() >= 16 && ds.display_rect.height() >= 16 &&
        ds.display_rect.right <= sw && ds.display_rect.bottom <= sh)
    {
        rx = ds.display_rect.left; ry = ds.display_rect.top;
        rw = ds.display_rect.width(); rh = ds.display_rect.height();
    }

    // No scaling blit is available on D3D11/D3D12 (only D3D9 and Vulkan can
    // rescale during a copy). So instead of shrinking the depth into the picture
    // rectangle, enlarge the CANVAS by the same proportion and copy 1:1 into the
    // matching spot. ReShade then stretches that canvas over the whole frame, and
    // the depth lands exactly on the picture.
    //
    //   canvas   = capture * screen / picture
    //   offset   = picture offset * capture / picture
    //
    // 4:3 in a 16:9 window: a 3840x2688 capture goes into a 5120x2688 canvas at
    // x=640, and stretching that back over 3840x2160 puts depth in the middle 75%.
    // Manual nudge, applied to the picture rectangle before anything is derived from
    // it. Scaling about the rectangle's own centre keeps it centred while it shrinks,
    // which is what "the overlay is a bit too wide" actually needs.
    if (g_cfg_place_scale != 1.0f || g_cfg_place_x != 0.0f || g_cfg_place_y != 0.0f)
    {
        const float cx = rx + rw * 0.5f, cy = ry + rh * 0.5f;
        float nw = rw * g_cfg_place_scale, nh = rh * g_cfg_place_scale;
        if (nw < 16.0f) nw = 16.0f;
        if (nh < 16.0f) nh = 16.0f;
        float nx = cx - nw * 0.5f + g_cfg_place_x * (float)sw;
        float ny = cy - nh * 0.5f + g_cfg_place_y * (float)sh;
        if (nx < 0.0f) nx = 0.0f;
        if (ny < 0.0f) ny = 0.0f;
        if (nx + nw > (float)sw) nw = (float)sw - nx;
        if (ny + nh > (float)sh) nh = (float)sh - ny;
        rx = (uint32_t)nx; ry = (uint32_t)ny;
        rw = (uint32_t)nw; rh = (uint32_t)nh;
    }

    // Auto: look for the black bars when something changed. What it finds is used from the next frame.
    AutoAspectTick(dev, runtime, ds, sw, sh);

    // Use the rendered region, not the whole texture. On a GameCube EFB the capture is
    // 640x528 while the picture is the top 640x480 of it; scaling the full texture onto
    // the picture stretches everything by 528/480 and slides it down the screen.
    const uint32_t capW = (ds.raw_vp_w && ds.raw_vp_w <= ds.raw_w) ? ds.raw_vp_w : ds.raw_w;
    const uint32_t capH = (ds.raw_vp_h && ds.raw_vp_h <= ds.raw_h) ? ds.raw_vp_h : ds.raw_h;

    uint64_t cw64 = (uint64_t)capW * sw / (rw ? rw : 1);
    uint64_t ch64 = (uint64_t)capH * sh / (rh ? rh : 1);
    if (cw64 > 16384) cw64 = 16384;
    if (ch64 > 16384) ch64 = 16384;
    uint32_t cw = (uint32_t)cw64, ch = (uint32_t)ch64;
    if (cw < capW) cw = capW;
    if (ch < capH) ch = capH;

    uint32_t ox = (uint32_t)((uint64_t)rx * capW / (rw ? rw : 1));
    uint32_t oy = (uint32_t)((uint64_t)ry * capH / (rh ? rh : 1));
    if (ox + capW > cw) ox = cw - capW;
    if (oy + capH > ch) oy = ch - capH;

    const uint32_t capX = (ds.raw_vp_x + capW <= ds.raw_w) ? ds.raw_vp_x : 0;
    const uint32_t capY = (ds.raw_vp_y + capH <= ds.raw_h) ? ds.raw_vp_y : 0;

    const bool passthrough = (cw == capW && ch == capH && capW == ds.raw_w && capH == ds.raw_h &&
                              capX == 0 && capY == 0);

    if (passthrough)
    {
        if (ds.disp != ds.raw || ds.disp_srv == 0)
        {
            if (ds.disp_srv != 0) { dev->destroy_resource_view(ds.disp_srv); ds.disp_srv = { 0 }; }
            if (ds.disp != 0 && ds.disp != ds.raw) { dev->destroy_resource(ds.disp); }
            ds.disp = ds.raw; ds.disp_w = ds.raw_w; ds.disp_h = ds.raw_h;
            ds.disp_fmt = format::unknown;
            dev->create_resource_view(ds.raw, resource_usage::shader_resource,
                resource_view_desc(DepthReadFormat(format_to_typeless(ds.raw_fmt))), &ds.disp_srv);
        }
        if (ds.disp_srv != 0)
        {
            const resource_view pub = Linearise(dev, runtime, ds, ds.disp, ds.disp_srv, ds.raw_w, ds.raw_h);
            runtime->update_texture_bindings("DEPTH", pub, pub);
        }
        sprintf_s(g_status, "%ux%u depth over the whole %ux%u frame", ds.raw_w, ds.raw_h, sw, sh);
        if (ds.logged_rect != 0)
        {
            ds.logged_rect = 0;
            Log("[depth-charge] full-frame picture: publishing the %ux%u capture directly", ds.raw_w, ds.raw_h);
        }
        return;
    }

    // (Oct 19 2026) The depth placed in the linearisation pass, not on a canvas. Dan's 5120x2160 ultrawide: PCSX2 shows its
    // 16:9 picture in the middle, so for the first time on PCSX2 a canvas was needed - and creating it failed every frame
    // ("could not create the 3413x1792 canvas"): a canvas has the capture's own format and, its margins cleared by drawing,
    // must be a render target, which PCSX2's depth (D32_FLOAT_S8X24, r32_g8_x24_typeless) can never be. On that failure
    // nothing at all was published, so every effect read depth 0 everywhere. The linearisation pass draws every texel of the
    // published depth anyway: it reads the capture through the placement and writes "nothing drawn" (far) in the margins.
    // The canvas below is only the fallback, for when linearisation is off or unavailable.
    resource_view pub = { 0 };
    bool placed = false;
    if (g_cfg_linearise && BuildConverter(dev, ds))
    {
        if (ds.disp != ds.raw || ds.disp_srv == 0)
        {
            if (ds.disp_srv != 0) { dev->destroy_resource_view(ds.disp_srv); ds.disp_srv = { 0 }; }
            if (ds.disp_rtv != 0) { dev->destroy_resource_view(ds.disp_rtv); ds.disp_rtv = { 0 }; }
            if (ds.disp != 0 && ds.disp != ds.raw) { dev->destroy_resource(ds.disp); }
            ds.disp = ds.raw; ds.disp_w = ds.raw_w; ds.disp_h = ds.raw_h;
            ds.disp_fmt = format::unknown;
            dev->create_resource_view(ds.raw, resource_usage::shader_resource,
                resource_view_desc(DepthReadFormat(format_to_typeless(ds.raw_fmt))), &ds.disp_srv);
        }
        if (ds.disp_srv == 0)
            return;
        const float iw = 1.0f / (float)ds.raw_w, ih = 1.0f / (float)ds.raw_h;
        const float place[8] = {
            (float)cw * iw, (float)ch * ih,                                  // published uv -> capture uv: scale,
            ((float)capX - (float)ox) * iw, ((float)capY - (float)oy) * ih,  // offset
            (float)capX * iw, (float)capY * ih,                              // the captured rectangle
            (float)(capX + capW) * iw, (float)(capY + capH) * ih,
        };
        pub = Linearise(dev, runtime, ds, ds.raw, ds.disp_srv, cw, ch, place);
        placed = (pub != ds.disp_srv);
        if (!placed)
        {
            // the linearisation failed after all: the capture as it is - unaligned, but not nothing
            runtime->update_texture_bindings("DEPTH", pub, pub);
            sprintf_s(g_status, "%ux%u depth over the whole %ux%u frame - UNALIGNED (the linearisation failed)", ds.raw_w, ds.raw_h, sw, sh);
            return;
        }
    }
    else
    {
    if (ds.disp == 0 || ds.disp == ds.raw || ds.disp_w != cw || ds.disp_h != ch ||
        ds.disp_fmt != format_to_typeless(ds.raw_fmt))
    {
        if (ds.disp_srv != 0) { dev->destroy_resource_view(ds.disp_srv); ds.disp_srv = { 0 }; }
        if (ds.disp != 0 && ds.disp != ds.raw) { dev->destroy_resource(ds.disp); ds.disp = { 0 }; }

        const format typeless = format_to_typeless(ds.raw_fmt);
        if (ds.disp_rtv != 0) { dev->destroy_resource_view(ds.disp_rtv); ds.disp_rtv = { 0 }; }
        resource_desc bd(cw, ch, 1, 1, typeless, 1, memory_heap::default_,
                         resource_usage::copy_dest | resource_usage::shader_resource |
                         resource_usage::render_target);
        bool can_draw = true;
        if (!dev->create_resource(bd, nullptr, resource_usage::shader_resource, &ds.disp))
        {
            // (Oct 19 2026) a depth format can't be a render target (PCSX2's r32_g8_x24): then without, margins not cleared
            bd.usage = resource_usage::copy_dest | resource_usage::shader_resource;
            can_draw = false;
            if (!dev->create_resource(bd, nullptr, resource_usage::shader_resource, &ds.disp))
            {
                Log("[depth-charge] could not create the %ux%u canvas; nothing published", cw, ch);
                ds.disp = { 0 }; return;
            }
        }
        if (!dev->create_resource_view(ds.disp, resource_usage::shader_resource,
                resource_view_desc(DepthReadFormat(typeless)), &ds.disp_srv))
        { dev->destroy_resource(ds.disp); ds.disp = { 0 }; return; }
        if (can_draw)
            dev->create_resource_view(ds.disp, resource_usage::render_target,
                    resource_view_desc(DepthReadFormat(typeless)), &ds.disp_rtv);
        ds.disp_w = cw; ds.disp_h = ch; ds.disp_fmt = typeless;
    }

    const subresource_box src_box = { capX, capY, 0, capX + capW, capY + capH, 1 };
    const subresource_box dst_box = { ox, oy, 0, ox + capW, oy + capH, 1 };
    const resource       res [1] = { ds.disp };
    const resource_usage from[1] = { resource_usage::shader_resource };
    const resource_usage to  [1] = { resource_usage::copy_dest };

    // Clear the margins rather than filling them.
    //
    // These strips exist because the canvas is wider than the picture, and nothing
    // ever draws into them. They used to be filled by copying the outermost slice of
    // the capture outwards, described as "behaves like clamp" - but clamp repeats one
    // EDGE COLUMN, while that copied a hundred-pixel band of the picture. The result
    // was a visible duplicate of the image down each side, which reads as a rendering
    // bug and hid the real placement while we were trying to judge it.
    //
    // Zero is the honest value: nothing was drawn there. The conversion already maps
    // raw 0 to the far plane, so consumers see empty distance instead of a ghost of
    // the scene, and the debug views show flat margins that are obviously not content.
    if (ds.disp_rtv != 0)
    {
        const resource_usage rt[1] = { resource_usage::render_target };
        cmd_list->barrier(1, res, from, rt);
        const float zero[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        cmd_list->clear_render_target_view(ds.disp_rtv, zero);
        cmd_list->barrier(1, res, rt, to);
    }
    else
    {
        cmd_list->barrier(1, res, from, to);
    }

    cmd_list->copy_texture_region(ds.raw, 0, &src_box, ds.disp, 0, &dst_box);   // 1:1, no scaling

    cmd_list->barrier(1, res, to, from);
    pub = Linearise(dev, runtime, ds, ds.disp, ds.disp_srv, cw, ch);
    }

    const int sig = (int)(ox * 100000 + cw) + (placed ? 1000000000 : 0);
    if (ds.logged_rect != sig)
    {
        ds.logged_rect = sig;
        Log("[depth-charge] picture %ux%u at %u,%u of a %ux%u frame -> %ux%u of a %ux%u capture "
                "placed at %u,%u on a %ux%u canvas - aspect corrected (%s)",
                rw, rh, rx, ry, sw, sh, capW, capH, ds.raw_w, ds.raw_h, ox, oy, cw, ch,
                placed ? "placed by the linearisation pass" : "copied onto a canvas");
    }

    if (rw > 0 && rh > 0)
    {
        const float a = (float)rw / (float)rh;
        if (a > 0.2f && a < 5.0f && fabsf(a - g_picture_aspect) > 0.001f)
        {
            g_picture_aspect = a;
            g_aspect_tries   = 1200;   // ~20s of frames to find the shaders
        }
    }

    sprintf_s(g_status, "picture %ux%u at %u,%u of %ux%u | %ux%u of a %ux%u capture on a %ux%u %s",
              rw, rh, rx, ry, sw, sh, capW, capH, ds.raw_w, ds.raw_h, cw, ch, placed ? "placement" : "canvas");
    runtime->update_texture_bindings("DEPTH", pub, pub);
}

static void draw_settings(effect_runtime *)
{
    ImGui::TextUnformatted("Select the game's aspect ratio.");
    ImGui::Spacing();

    if (ImGui::Combo("Picture aspect", &g_aspect_index, kAspectLabel, 6))
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_cfg_aspect = kAspectValue[g_aspect_index];
        if (g_aspect_index == 0)
            g_auto_repick = true;     // switching to Auto makes it look for the black bars
        for (auto &kv : g_devices) kv.second.logged_rect = -1;
        MarkUserProfile();
        WriteConfig();
        Log("[depth-charge] picture aspect set to %s from the overlay", kAspectText[g_aspect_index]);
    }

    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Set this to match the aspect the emulator is displaying.\n"
                          "PCSX2 4:3 in a 16:9 window pillarboxes the picture, so depth\n"
                          "has to be inset by the same amount or it lands off the geometry.\n"
                          "Saved to depth-charge.cfg.");

    if (g_aspect_index == 0 && g_cfg_auto_bars)
    {
        char st[128];
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            strcpy_s(st, g_auto_status);
        }
        if (ImGui::Button("Check again"))
            g_auto_repick = true;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Look for the black bars around the picture again.\n"
                              "Press this after changing the emulator's own aspect setting.");
        ImGui::SameLine();
        ImGui::TextDisabled("%s", st);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Auto finds the picture from the black bars around it. It looks when the\n"
                              "game starts, when the window changes size, and when you pick Auto.\n"
                              "If you change the emulator's own aspect setting, press Check again.");
    }

    ImGui::Separator();
    ImGui::TextUnformatted("Depth conversion");

    if (ImGui::Checkbox("Convert depth to real distance", &g_cfg_linearise))
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        WriteConfig();
        Log("[depth-charge] linearise set to %d from the overlay", g_cfg_linearise ? 1 : 0);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("A depth buffer is  raw = A + B/distance.  A is a large constant the\n"
                          "distance term rides on; on emulated content it swamps the signal, so\n"
                          "anything that does not subtract it first comes out flat. This removes\n"
                          "it and converts to real linear distance before publishing, so every\n"
                          "ReShade effect gets correct depth with no shader edits.\n\n"
                          "ReShade's own depth settings can stay as they are: the depth is written\n"
                          "so that they read it back as this distance.");

    if (g_cfg_linearise)
    {
        if (ImGui::Checkbox("Depth is already a distance (PS1 / DuckStation)", &g_cfg_linear_buffer))
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            MarkUserProfile();
            WriteConfig();
            Log("[depth-charge] linear_buffer set to %d from the overlay", g_cfg_linear_buffer ? 1 : 0);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Tick this when the emulator's depth is ALREADY distance rather than 1/z.\n\n"
                "A PS2 or GameCube buffer is  raw = A + B/distance,  with a large constant A\n"
                "that swamps the signal - removing it and taking the reciprocal is the whole\n"
                "point of this tool there.\n\n"
                "DuckStation's PGXP depth is not that. It stores the GTE's own Z divided by\n"
                "65535, so the value IS normalised view distance: small near, large far.\n"
                "Running the 1/z conversion over it turns correct linear depth INTO a\n"
                "hyperbolic curve - the near and far ends swallow the range and the middle\n"
                "collapses, which looks like reflections and lighting that only work in a\n"
                "band and slide about when you touch the zero point.\n\n"
                "With this ticked, Depth Finder solves only a scale: no asymptote, no\n"
                "reciprocal, no flip. Leave it OFF for PS2 and GameCube.\n\n"
                "Saved per game.");
    }

    if (g_cfg_linearise)
    {
        // Which disc this is, and whether anything has ever been measured for it.
        // Without this the sliders look authoritative no matter what is running.
        if (g_game_key.empty())
            ImGui::TextDisabled("No game detected in the window title - settings are not per-game here");
        else if (g_game_source == SRC_USER)
            ImGui::TextDisabled("%s - your measured settings", g_game_key.c_str());
        else if (g_game_source == SRC_BUILTIN)
            ImGui::TextDisabled("%s - tested defaults", g_game_key.c_str());
        else
            ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.2f, 1.0f),
                               "%s has not been measured. Press Depth Finder.", g_game_key.c_str());

        // Always available when a game is in the shipped table, whatever the player
        // has done to it since. Getting back to a known-good state has to be one
        // click, or nobody will risk experimenting in the first place.
        if (const BuiltinProfile *b = FindBuiltin(g_game_key))
        {
            if (g_game_source == SRC_USER)
            {
                const float rw = ImGui::CalcTextSize("Reset to tested defaults").x
                               + ImGui::GetStyle().FramePadding.x * 6.0f;
                if (ImGui::Button("Reset to tested defaults", ImVec2(rw, 0)))
                {
                    ApplyBuiltin(*b);
                    g_profiles.erase(g_game_key);       // drop the override entirely
                    g_game_source = SRC_BUILTIN;
                    g_cal_status[0] = 0;
                    WriteConfig();
                    Log("[depth-charge] \"%s\" reset to tested defaults", g_game_key.c_str());
                }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Throw away your changes for this game and go back to the\n"
                                      "values it was tested with. Other games are untouched.");
            }
        }

        if (g_cal_state == CAL_IDLE)
        {
            // Window titles only change when an emulator loads a different game, so this is a
            // button rather than a poll. Asking a window for its title is a blocking
            // cross-thread message; doing it twice a second cost us a whole evening.
            const float btnW = ImGui::CalcTextSize("Depth Finder").x
                             + ImGui::GetStyle().FramePadding.x * 6.0f;
            // Sits beside Depth Finder rather than spanning the panel - it is an occasional
            // action (only emulators rename their window), not a headline control.
            if (ImGui::Button("Check game", ImVec2(btnW, 0)))
                g_title_recheck = true;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Press this after switching games.\n"
                                  "(Reads the window title and switches to that game's profile).");
            ImGui::SameLine();
            if (ImGui::Button("Depth Finder", ImVec2(btnW, 0)))
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                g_cal_mins.clear(); g_cal_maxs.clear(); g_cal_huds.clear();
                g_cal_frames = 0; g_cal_ok = false; g_cal_status[0] = 0;
                g_cal_flip_votes = 0; g_cal_vote_frames = 0;
                g_dbg_view_before = g_dbg_view;
                g_cal_started = (double)GetTickCount64() / 1000.0;
                g_cal_state = CAL_CAPTURING;
                Log("[depth-charge] calibration started");
            }
            ImGui::TextWrapped("Auto detects depth. Look toward a long sightline, press the "
                               "button, and hold still. The resulting debug overlay should "
                               "appear as a gradient from blue (close) to red (far).");
        }
        else if (g_cal_state == CAL_CAPTURING)
        {
            const double now = (double)GetTickCount64() / 1000.0;
            const float t = (float)((now - g_cal_started) / g_cal_seconds);
            ImGui::ProgressBar(t, ImVec2(-1, 0), "measuring - hold still");
        }
        else
        {
            ImGui::TextWrapped("%s", g_cal_status);
        }

        if (g_cal_status[0] != 0 && g_cal_state == CAL_IDLE)
            ImGui::TextWrapped("%s", g_cal_status);

        ImGui::Spacing();
        bool changed = false;
        changed |= ImGui::DragFloat("Zero point (log2)", &g_cfg_zero_log2, 0.005f, -40.0f, 0.0f, "%.3f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Where the buffer sits at infinite distance. This is the one that\n"
                              "matters and it is sharp - the useful signal can be only a few\n"
                              "percent wide, so move it in single steps.\n"
                              "Too high and the far field clips to one flat value; too low and\n"
                              "the whole scene bunches up near the camera.\n"
                              "Measure it per game with the Depth Charge Inspector's log view.");
        changed |= ImGui::DragFloat("Far distance", &g_cfg_far_scale, 1000.0f, 1.0f, 1e9f, "%.0f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Which distance reads as 1.0. Set the zero point first -\n"
                              "this only rescales what that exposes.");

        float ui_cut = g_cfg_overlay_cutoff < 0.0f ? 0.0f : g_cfg_overlay_cutoff;
        if (ImGui::DragFloat("HUD cutoff", &ui_cut, 0.0002f, 0.0f, 0.2f, "%.4f"))
        {
            g_cfg_overlay_cutoff = ui_cut;              // touching it makes it managed
            g_cutoff_tries = 1200;
            changed = true;
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Depth below which a pixel is a 2D overlay - HUD, menus, subtitles -\n"
                              "rather than geometry. It is sent to an OVERLAY_NEAR_CUTOFF setting in\n"
                              "lumenite_Kernel.fx so nothing shades the HUD. Stock LumeniteFX has\n"
                              "no such setting yet, so there this does nothing.\n\n"
                              "Depth Finder sets this from the same measurement as the two above:\n"
                              "it already has to separate the HUD from geometry to keep it out of\n"
                              "the near end, so it knows where both sit. It lands halfway between\n"
                              "them on a log scale.\n\n"
                              "This is in depth units, so re-calibrating moves it - which is why it\n"
                              "is derived here rather than left as a number to tune by hand.\n\n"
                              "0 = leave Lumenite's own slider alone.");
        if (g_cutoff_target == 0)
            ImGui::TextDisabled("No loaded effect has a HUD cutoff setting - this does nothing here");
        if (g_cal_hud_depth > 0.0f)
            ImGui::TextDisabled("HUD measured at %.5f, nearest geometry at %.5f",
                                g_cal_hud_depth, g_cal_geo_depth);
        else if (g_cfg_overlay_cutoff == 0.0f)
            ImGui::TextDisabled("HUD and geometry could not be separated - exclusion off");
        else if (g_cfg_overlay_cutoff < 0.0f)
            ImGui::TextDisabled("not measured - Lumenite's own setting is left alone");

        // What the measurement saw, in raw buffer units. Everything above is derived
        // from these two numbers, so when a result looks wrong this is what says why.
        if (g_cal_raw_near > 0.0f)
        {
            // A 1/z buffer has the larger value up close; a linear one has it far away.
            const float lo = (g_cal_raw_far < g_cal_raw_near) ? g_cal_raw_far : g_cal_raw_near;
            const float hi = (g_cal_raw_far < g_cal_raw_near) ? g_cal_raw_near : g_cal_raw_far;
            ImGui::TextDisabled("measured raw  far 2^%.4f  near 2^%.4f  (span %.2f%%)",
                                log2f(g_cal_raw_far), log2f(g_cal_raw_near),
                                100.0f * (hi - lo) / ((lo > 0.0f) ? lo : 1.0f));
        }

        if (ImGui::Checkbox("Snap zero point to a power of two", &g_cfg_snap_zero))
            changed = true;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Console games build their Z scale out of integer ranges, so the\n"
                              "asymptote usually sits exactly on a power of two. When the solve\n"
                              "lands within 2%% of one, this takes the power of two - which makes\n"
                              "the far scale reproducible instead of drifting run to run.\n\n"
                              "Turn it off if a game's depth is not built that way. Symptom: the\n"
                              "map comes back flat or saturated even though the RAW debug view\n"
                              "shows a proper picture.");

        if (changed)
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            MarkUserProfile();
            WriteConfig();
        }
    }

    ImGui::Separator();
    if (ImGui::CollapsingHeader("Inspector"))
    {
        ImGui::TextWrapped("Debug views drawn over the finished frame. Everything here comes "
                           "from the depth buffer alone, so it works in any game.");
        ImGui::Spacing();

        const char *kViews =
            "Off\0"
            "Linear depth\0"
            "Linear depth - false colour\0"
            "RAW on a LOG scale\0"
            "Near-field zoom\0"
            "Depth quantisation bands\0"
            "Reconstructed normals\0"
            "Flatness mask\0"
            "Overlay (HUD) rejection band\0";
        const int view_was = g_dbg_view;
        ImGui::Combo("Debug view", &g_dbg_view, kViews);
        // Opening the overlay band view used to show the slider's own leftover value,
        // which reads exactly like the live setting and is not. Seed it from the real
        // one so what you see on screen is what the effects are actually doing.
        if (g_dbg_view == 8 && view_was != 8)
            g_dbg_hud_cut = (g_cfg_overlay_cutoff >= 0.0f) ? g_cfg_overlay_cutoff : 0.0f;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("False colour is the sanity check - a correct depth curve sweeps\n"
                              "smoothly from blue nearby to red in the distance.\n"
                              "The LOG view is the measuring instrument: squeeze its two ends\n"
                              "until the scene fills the ramp, and the numbers you read off are\n"
                              "the zero point and range for this game.");

        if (g_dbg_view == 3)
        {
            ImGui::DragFloat("Log range: far end",  &g_dbg_log_min, 0.05f, -24.0f, 0.0f, "%.2f");
            ImGui::DragFloat("Log range: near end", &g_dbg_log_max, 0.05f, -24.0f, 0.0f, "%.2f");
        }
        if (g_dbg_view == 4)
            ImGui::DragFloat("Near-field gain", &g_dbg_near_gain, 10.0f, 1.0f, 1000000.0f, "%.0f");
        if (g_dbg_view == 5)
            ImGui::DragFloat("Band frequency", &g_dbg_bands, 16.0f, 16.0f, 8192.0f, "%.0f");
        if (g_dbg_view == 6 || g_dbg_view == 7)
            ImGui::DragFloat("Assumed field of view", &g_dbg_fov, 1.0f, 30.0f, 120.0f, "%.0f");
        if (g_dbg_view == 8)
        {
            if (g_cfg_overlay_cutoff < 0.0f)
                ImGui::TextUnformatted("In use right now: none (never measured)");
            else
                ImGui::Text("In use right now: %.4f", g_cfg_overlay_cutoff);

            ImGui::DragFloat("Try a cutoff", &g_dbg_hud_cut, 0.0005f, 0.0f, 0.2f, "%.4f");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("This slider only changes the picture below, not the game.\n"
                                  "Yellow is what a cutoff of this value would treat as a 2D\n"
                                  "overlay: HUD, menus, lens flares. It should cover those and\n"
                                  "nothing else. Press Use this value to make it real.\n\n"
                                  "Check it in gameplay, not paused, since a menu hides the near\n"
                                  "ground, which is exactly what a too-high cutoff eats.");

            const float uw = ImGui::CalcTextSize("Use this value").x
                           + ImGui::GetStyle().FramePadding.x * 6.0f;
            if (ImGui::Button("Use this value", ImVec2(uw, 0)))
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                g_cfg_overlay_cutoff = g_dbg_hud_cut;
                g_cutoff_tries = 8;          // let it push into the shaders again
                MarkUserProfile();
                WriteConfig();
                Log("[depth-charge] overlay cutoff set to %.5f from the overlay",
                    g_cfg_overlay_cutoff);
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Applies the trial value above as the real cutoff and saves it\n"
                                  "for this game.");
        }
    }

    ImGui::Separator();
    if (!ImGui::CollapsingHeader("Advanced"))
    {
        ImGui::Spacing();
        ImGui::TextUnformatted("Current placement:");
        ImGui::TextWrapped("%s", g_status);
        return;
    }

    if (ImGui::DragFloat("Depth curve", &g_cfg_depth_curve, 0.01f, 0.25f, 4.0f, "%.2f"))
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_cfg_depth_curve < 0.25f) g_cfg_depth_curve = 0.25f;
        if (g_cfg_depth_curve > 4.00f) g_cfg_depth_curve = 4.00f;
        MarkUserProfile();
        WriteConfig();
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "The shape of the depth between the near and far anchors. Both ends stay\n"
            "put; only the middle moves.\n\n"
            "1.00 is physically correct - proportional to true distance. Higher values\n"
            "lean back toward the raw buffer's own curve, which is hyperbolic: stretched\n"
            "up close, compressed at distance.\n\n"
            "Correct is not always what the effects were built for. Every ReShade effect\n"
            "was written against PC depth buffers, which are hyperbolic too, so some of\n"
            "them read a linear buffer as subtly wrong-shaped geometry. If turning\n"
            "linearisation OFF looks better than leaving it on, the difference is this\n"
            "curve and nothing else - so find the value that looks right rather than\n"
            "choosing between two extremes.\n\n"
            "Saved per game.");

    ImGui::TextUnformatted("Placement");
    bool nudged = false;
    nudged |= ImGui::DragFloat("Placement size",  &g_cfg_place_scale, 0.002f, 0.50f, 1.50f, "%.3f");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Shrink or grow the depth overlay to match the picture.\n\n"
                          "Needed when an emulator draws its final picture with a full-frame\n"
                          "viewport, so nothing in the graphics stream says where the picture\n"
                          "actually is. Dolphin does this. Turn on a debug view and size the\n"
                          "overlay onto the game by eye. Saved per game.");
    nudged |= ImGui::DragFloat("Placement X",     &g_cfg_place_x, 0.001f, -0.30f, 0.30f, "%.3f");
    nudged |= ImGui::DragFloat("Placement Y",     &g_cfg_place_y, 0.001f, -0.30f, 0.30f, "%.3f");
    if (nudged)
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        for (auto &kv : g_devices) kv.second.logged_rect = -1;
        MarkUserProfile();
        WriteConfig();
    }

    {
        const float sw = ImGui::CalcTextSize("Survey depth passes").x
                       + ImGui::GetStyle().FramePadding.x * 6.0f;
        if (ImGui::Button("Survey depth passes", ImVec2(sw, 0)))
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_survey_frames = 4;
            g_survey_binds  = 0;
            Log("[depth-charge] Survey start: every depth pass of the next frame");
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Writes every depth target drawn during the next single frame to\n"
                              "the ReShade log - size, format and draw count - and marks which\n"
                              "ones clear the draw-count filter.\n\n"
                              "Use it when the captured depth looks like the wrong buffer: it\n"
                              "shows what the chooser was choosing between.");
    }

    if (ImGui::Checkbox("Keep effects off the HUD", &g_cfg_hud_split))
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        for (auto &kv : g_devices)
        {
            kv.second.hud_index = -1; kv.second.hud_next = -1; kv.second.hud_best = 0;
            kv.second.co_index = 0;   kv.second.hud_done = false; kv.second.hud_inside = false;
            kv.second.hud_frames = 0;
        }
        WriteConfig();
        Log("[depth-charge] effects_before_hud set to %d from the overlay", g_cfg_hud_split ? 1 : 0);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Stops reflections, ambient occlusion and DLSS 5 from being applied\n"
                          "to the health bars, ammo counter and other on-screen furniture.\n\n"
                          "A HUD is painted straight into the picture with no depth of its own,\n"
                          "so at those pixels the depth belongs to whatever is behind them. No\n"
                          "cutoff or threshold can find it. The only real fix is to run the\n"
                          "effects EARLIER, just before the game draws its HUD, so the HUD lands\n"
                          "on top of a finished image instead of being treated as scenery.\n\n"
                          "Depth Charge finds that moment by watching the frame: the HUD is the\n"
                          "full-screen pass with no depth attached that draws the most, and it is\n"
                          "the same pass every frame, so the one seen last frame is used for the\n"
                          "next. Nothing is assumed about the game.\n\n"
                          "Emulators do not need this. Their HUD belongs to the emulated game and\n"
                          "is in the depth buffer already, where the overlay cutoff handles it.\n"
                          "Saved to depth-charge.cfg.");
    if (g_cfg_hud_split)
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        const device_state *ds = LiveDevice();
        if (ds == nullptr)
            ImGui::TextUnformatted("Waiting for the game to present a frame.");
        else if (ds->hud_index >= 0)
            ImGui::Text("HUD found: full-screen pass %d, split on %llu frames",
                        ds->hud_index, (unsigned long long)ds->hud_frames);
        else
            ImGui::Text("Looking for the HUD pass. Last frame had %d full-screen passes "
                        "with no depth; busiest drew %u, and %u is the minimum.",
                        ds->co_seen, ds->hud_seen_best, g_cfg_hud_min_draws);
    }

    if (ImGui::Checkbox("Use the whole scene, not the busiest pass", &g_cfg_whole_scene))
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        for (auto &kv : g_devices) { kv.second.iv_target = 0; kv.second.iv_draws = 0; kv.second.iv_best = 0; }
        WriteConfig();
        Log("[depth-charge] whole_scene set to %d from the overlay", g_cfg_whole_scene ? 1 : 0);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Turn this on if the depth goes solid red or freezes when you face a\n"
                          "wall, look at the floor, or spin on the spot.\n\n"
                          "Those are the moments when very little of the level is on screen.\n"
                          "A game like BioShock draws the world, wipes the depth buffer, then\n"
                          "draws the gun in your hands into the same buffer so it can never\n"
                          "poke through a wall. The gun is the same handful of draws every\n"
                          "frame; the level is not. Picking whichever pass drew the most, the\n"
                          "normal rule, therefore picks the gun the moment the level gets\n"
                          "simple, and the effects are handed a picture of a gun.\n\n"
                          "With this on, depth is taken at each wipe instead, and the last\n"
                          "stretch to finish before the frame is shown wins. That is the level,\n"
                          "whatever you are looking at, because the gun is always drawn after\n"
                          "it. And it is the complete level rather than one pass of it.\n\n"
                          "Leave it off for emulators. They do not wipe depth mid-frame, so\n"
                          "there is no boundary to measure and nothing would be captured.\n"
                          "Saved to depth-charge.cfg.");

    if (ImGui::Checkbox("Merge every pass (fixes holes)", &g_cfg_accumulate))
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        MarkUserProfile();          // merging is a per-game property, so it saves per game
        WriteConfig();
        Log("[depth-charge] accumulate set to %d from the overlay", g_cfg_accumulate ? 1 : 0);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("PCSX2 draws a frame across many passes and depth accumulates\n"
                          "across them. Capturing only the busiest pass throws away\n"
                          "everything drawn afterwards - typically the player model and\n"
                          "the nearest ground, which come out as holes with no depth.\n"
                          "Leave this on unless you are comparing against the old behaviour.\n"
                          "Saved to depth-charge.cfg.");

    ImGui::Spacing();
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        for (auto &kv : g_devices)
        {
            ImGui::Text("Passes merged last frame: %u", kv.second.last_merged);
            break;
        }
    }
    ImGui::TextUnformatted("Current placement:");
    ImGui::TextWrapped("%s", g_status);
}

// events

// Lifetime of each list's cmd_state. Subscribing to these changes how ReShade wraps command lists,
// whatever the callbacks do (see DllMain), so they have their own gate: with track_cmd_lifetime=0
// the race is still fixed (CmdState creates on first use) and only the cleanup is lost.
static void on_init_command_list(command_list *cmd_list)
{
    CmdState(cmd_list);
}

static void on_destroy_command_list(command_list *cmd_list)
{
    cmd_list->destroy_private_data<cmd_state>();
}

static void on_bind_viewports(command_list *cmd_list, uint32_t first, uint32_t count, const viewport *viewports)
{
    if (!g_cfg_track_binds) return;
    if (count == 0 || first != 0 || viewports == nullptr) return;
    cmd_state &cs = CmdState(cmd_list);
    cs.vp = viewports[0];
    cs.vp_set = true;
    if (cs.dsv != 0 && viewports[0].width * viewports[0].height > cs.vp_max_w * cs.vp_max_h)
    {
        cs.vp_max_w = viewports[0].width;
        cs.vp_max_h = viewports[0].height;
    }
}

static void on_bind_scissors(command_list *cmd_list, uint32_t first, uint32_t count, const rect *rects)
{
    if (!g_cfg_track_binds) return;
    if (count == 0 || first != 0 || rects == nullptr) return;
    device *const dev = cmd_list->get_device();
    std::lock_guard<std::mutex> lock(g_mutex);
    device_state &ds = g_devices[dev];
    ds.last_scissor = rects[0];
    ds.scissor_set = true;
    cmd_state &cs = CmdState(cmd_list);
    cs.sc = rects[0];
    cs.sc_set = true;
}

static void on_bind_rtv_dsv(command_list *cmd_list, uint32_t count, const resource_view *rtvs, resource_view dsv)
{
    if (!g_cfg_track_binds) return;
    cmd_state &cs = CmdState(cmd_list);

    {
        device *const d0 = cmd_list->get_device();
        std::lock_guard<std::mutex> lock(g_mutex);
        device_state &ds0 = g_devices[d0];
        ++ds0.rt_binds;
        if (dsv != 0) ++ds0.dsv_binds;
    }

    if (g_survey_frames > 0 && g_survey_binds < 400)
    {
        ++g_survey_binds;
        device *const d = cmd_list->get_device();
        char rt[256]; int n = 0;
        rt[0] = 0;
        for (uint32_t i = 0; i < count && i < 4 && rtvs != nullptr; ++i)
        {
            if (rtvs[i] == 0) continue;
            const resource r = d->get_resource_from_view(rtvs[i]);
            if (r == 0) continue;
            const resource_desc rd = d->get_resource_desc(r);
            n += sprintf_s(rt + n, sizeof(rt) - n, " rt%u=%ux%u fmt%u",
                           i, rd.texture.width, rd.texture.height, (unsigned)rd.texture.format);
        }
        if (dsv != 0)
        {
            const resource r = d->get_resource_from_view(dsv);
            const resource_desc rd = d->get_resource_desc(r);
            Log("[depth-charge] SURVEY bind: %u colour target(s)%s | DEPTH %ux%u fmt%u   (previous pass drew %u)",
                count, rt, rd.texture.width, rd.texture.height, (unsigned)rd.texture.format, cs.draws);
        }
        else
        {
            Log("[depth-charge] SURVEY bind: %u colour target(s)%s | no depth-stencil   (previous pass drew %u)",
                count, rt, cs.draws);
        }
    }

    // Is this the pass that draws the finished picture into the window? If so its
    // viewport is the rectangle the depth has to be placed into.
    if (count > 0 && rtvs != nullptr && cs.vp_set)
    {
        device *const dev = cmd_list->get_device();
        std::lock_guard<std::mutex> lock(g_mutex);
        device_state &ds = g_devices[dev];
        if (!ds.back_buffers.empty())
        {
            for (uint32_t i = 0; i < count; ++i)
            {
                if (rtvs[i] == 0) continue;
                const resource rt = dev->get_resource_from_view(rtvs[i]);
                // A handle in back_buffers is not proof: the set is filled when the
                // swapchain is created and never revalidated, and a driver happily
                // recycles a destroyed resource's handle for something else. Dolphin
                // recreates its swapchain (960x720 then 3840x2054 here) and the EFB
                // then inherits an old back-buffer handle, so its 3840x2880 render
                // pass was being read as the final blit - which is a rectangle taller
                // than the frame, which the Auto path then rejects, which is why Auto
                // silently fell back to stretching over the whole window.
                //
                // The picture cannot be larger than the thing it is drawn into.
                const bool fits = (ds.bb_w == 0 || ds.bb_h == 0) ||
                                  (cs.vp.width  <= (float)ds.bb_w + 1.0f &&
                                   cs.vp.height <= (float)ds.bb_h + 1.0f);
                if (rt != 0 && fits && ds.back_buffers.count(rt.handle))
                {
                    subresource_box b = {};
                    b.left = (uint32_t)(cs.vp.x < 0.0f ? 0.0f : cs.vp.x);
                    b.top  = (uint32_t)(cs.vp.y < 0.0f ? 0.0f : cs.vp.y);
                    b.right = b.left + (uint32_t)cs.vp.width;
                    b.bottom = b.top + (uint32_t)cs.vp.height;
                    b.back = 1;
                    ds.display_rect = b;
                    ds.display_rect_set = true;
                    ds.bb_binds++;
                    if (ds.vp_logs < 6)
                    {
                        ds.vp_logs++;
                        Log("[depth-charge] DIAG back-buffer draw #%u: viewport %.0fx%.0f at %.0f,%.0f | scissor %s%d,%d..%d,%d",
                            ds.bb_binds, cs.vp.width, cs.vp.height, cs.vp.x, cs.vp.y,
                            ds.scissor_set ? "" : "(none) ",
                            ds.scissor_set ? ds.last_scissor.left : 0, ds.scissor_set ? ds.last_scissor.top : 0,
                            ds.scissor_set ? ds.last_scissor.right : 0, ds.scissor_set ? ds.last_scissor.bottom : 0);
                    }
                    break;
                }
            }
        }
    }

    // When there is no depth-stencil at all, look for a colour target shaped like
    // depth. This is the only way to reach a PS1 emulator's reconstructed depth.
    resource_view eff = dsv;
    bool from_color = false;
    if (eff == 0 && rtvs != nullptr)
    {
        device *const d = cmd_list->get_device();
        for (uint32_t i = 0; i < count; ++i)
        {
            if (rtvs[i] == 0) continue;
            const resource r = d->get_resource_from_view(rtvs[i]);
            if (r == 0) continue;
            const resource_desc rd = d->get_resource_desc(r);
            if (rd.type == resource_type::texture_2d && LooksLikeDepthColour(rd.texture.format))
            {
                eff = rtvs[i]; from_color = true; break;
            }
        }
    }

    if (g_cfg_hud_split && eff == 0 && count > 0 && rtvs != nullptr && rtvs[0] != 0)
    {
        device *const d = cmd_list->get_device();
        const resource r = d->get_resource_from_view(rtvs[0]);
        resource_desc rd = {};
        if (r != 0) rd = d->get_resource_desc(r);

        effect_runtime *rt = nullptr;
        bool fire = false;
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            device_state &ds = g_devices[d];
            const bool frame_sized = ds.bb_w != 0 && rd.texture.width == ds.bb_w
                                                 && rd.texture.height == ds.bb_h;
            if (frame_sized)
            {
                // The pass that just ended was the previous colour-only one. Score it.
                if (cs.dsv == 0 && cs.draws > ds.hud_seen_best)
                    ds.hud_seen_best = cs.draws;
                if (cs.dsv == 0 && cs.draws > ds.hud_best && cs.draws >= g_cfg_hud_min_draws)
                {
                    ds.hud_best = cs.draws;
                    ds.hud_next = ds.co_index - 1;
                }
                if (!ds.hud_done && !ds.hud_inside && ds.hud_index >= 0
                    && ds.co_index == ds.hud_index && ds.runtime != nullptr)
                {
                    rt = ds.runtime;
                    ds.hud_inside = true;
                    fire = true;
                }
                ++ds.co_index;
            }
        }
        // Outside the lock: render_effects re-enters this add-on through the effects
        // events, which take the same mutex.
        if (fire && rt != nullptr)
        {
            rt->render_effects(cmd_list, rtvs[0], rtvs[0]);
            std::lock_guard<std::mutex> lock(g_mutex);
            device_state &ds = g_devices[d];
            ds.hud_inside = false;
            ds.hud_done   = true;
            ++ds.hud_frames;
        }
    }

    if (cs.dsv != eff)
    {
        if (cs.dsv != 0) snapshot(cmd_list, cs.dsv, cs.draws, cs.dsv_is_color);
        cs.dsv = eff;
        cs.dsv_is_color = from_color;
        cs.draws = 0;
        cs.vp_max_w = 0.0f; cs.vp_max_h = 0.0f;
        cs.sc_max_w = 0; cs.sc_max_h = 0; cs.sc_max_x = 0; cs.sc_max_y = 0;
    }
}

static bool on_begin_render_pass(command_list *cmd_list, uint32_t count, const render_pass_render_target_desc *rts,
                                 const render_pass_depth_stencil_desc *ds, render_pass_flags)
{
    if (!g_cfg_track_passes) return false;
    cmd_state &cs = CmdState(cmd_list);

    {
        device *const d0 = cmd_list->get_device();
        std::lock_guard<std::mutex> lock(g_mutex);
        device_state &ds0 = g_devices[d0];
        ++ds0.rp_begins;
        if (ds != nullptr && ds->view != 0) ++ds0.rp_depth;

        if (g_survey_frames > 0 && g_survey_binds < 400)
        {
            ++g_survey_binds;
            char rt[256]; int n = 0; rt[0] = 0;
            for (uint32_t i = 0; i < count && i < 4 && rts != nullptr; ++i)
            {
                if (rts[i].view == 0) continue;
                const resource r = d0->get_resource_from_view(rts[i].view);
                if (r == 0) continue;
                const resource_desc rd = d0->get_resource_desc(r);
                n += sprintf_s(rt + n, sizeof(rt) - n, " rt%u=%ux%u fmt%u",
                               i, rd.texture.width, rd.texture.height, (unsigned)rd.texture.format);
            }
            if (ds != nullptr && ds->view != 0)
            {
                const resource r = d0->get_resource_from_view(ds->view);
                const resource_desc rd = d0->get_resource_desc(r);
                Log("[depth-charge] SURVEY pass: %u colour target(s)%s | DEPTH %ux%u fmt%u",
                    count, rt, rd.texture.width, rd.texture.height, (unsigned)rd.texture.format);
            }
            else
            {
                Log("[depth-charge] SURVEY pass: %u colour target(s)%s | no depth attachment", count, rt);
            }
        }
    }

    if (cs.dsv != 0) snapshot(cmd_list, cs.dsv, cs.draws, cs.dsv_is_color);
    cs.dsv = (ds != nullptr) ? ds->view : resource_view { 0 };
    cs.dsv_is_color = false;
    cs.draws = 0;
    cs.vp_max_w = 0.0f; cs.vp_max_h = 0.0f;
    cs.sc_max_w = 0; cs.sc_max_h = 0; cs.sc_max_x = 0; cs.sc_max_y = 0;
    return false;
}

static bool on_end_render_pass(command_list *cmd_list)
{
    if (!g_cfg_track_passes) return false;
    cmd_state &cs = CmdState(cmd_list);
    snapshot(cmd_list, cs.dsv, cs.draws, cs.dsv_is_color);
    cs.dsv = { 0 }; cs.dsv_is_color = false; cs.draws = 0;
    cs.vp_max_w = 0.0f; cs.vp_max_h = 0.0f;
    cs.sc_max_w = 0; cs.sc_max_h = 0; cs.sc_max_x = 0; cs.sc_max_y = 0;
    return false;
}

// The viewport a draw actually used. Binding order is not guaranteed: Dolphin sets
// the viewport and THEN the render targets, so a viewport recorded only at bind time
// is thrown away by the pass-start reset and the pass looks smaller than it is.
// Reading it at draw time is order-independent.
static std::atomic<uint64_t> g_draw_total{0};

static inline void note_draw_viewport(cmd_state &cs)
{
    ++g_draw_total;
    if (cs.dsv == 0) return;
    if (cs.vp_set && cs.vp.width * cs.vp.height > cs.vp_max_w * cs.vp_max_h)
    {
        cs.vp_max_w = cs.vp.width;
        cs.vp_max_h = cs.vp.height;
    }
    if (cs.sc_set)
    {
        const int32_t w = cs.sc.right - cs.sc.left, h = cs.sc.bottom - cs.sc.top;
        if (w > 0 && h > 0 && (uint64_t)w * h > (uint64_t)cs.sc_max_w * cs.sc_max_h)
        {
            cs.sc_max_x = cs.sc.left; cs.sc_max_y = cs.sc.top;
            cs.sc_max_w = (uint32_t)w; cs.sc_max_h = (uint32_t)h;
        }
    }
}

static bool on_draw(command_list *cmd_list, uint32_t, uint32_t inst, uint32_t, uint32_t)
{ if (!g_cfg_count_draws) return false; cmd_state &cs = CmdState(cmd_list); cs.draws += (inst ? inst : 1); note_draw_viewport(cs); return false; }

static bool on_draw_indexed(command_list *cmd_list, uint32_t, uint32_t inst, uint32_t, int32_t, uint32_t)
{ if (!g_cfg_count_draws) return false; cmd_state &cs = CmdState(cmd_list); cs.draws += (inst ? inst : 1); note_draw_viewport(cs); return false; }

// Indirect draws never reach the draw / draw_indexed events, so a game that issues
// most of its geometry through DrawIndirect looks to us like it barely draws at all.
// That starves the draw filter: no pass ever clears it, the auto-relaxation walks
// g_min_draws down to 1, and we latch onto a tiny utility pass and snapshot it
// thousands of times. BioShock Remastered showed 3871 depth binds against 218
// counted draws over 600 frames, which is the signature of exactly this.
static bool on_clear_dsv(command_list *cmd_list, resource_view dsv, const float *, const uint8_t *,
                         uint32_t, const rect *)
{
    // A survey lists every depth pass of a frame, but a list of draw counts alone
    // cannot say which passes belong to the same picture. A clear is the boundary:
    // passes after it are drawing a fresh scene into the same buffer, so the depth
    // that reaches the screen is whatever was written since the LAST clear.
    if (g_survey_frames > 0 && dsv != 0)
    {
        device *const dev = cmd_list->get_device();
        const resource src = dev->get_resource_from_view(dsv);
        if (src != 0)
        {
            const resource_desc d = dev->get_resource_desc(src);
            Log("[depth-charge]   CLEAR %ux%u  target %llu",
                d.texture.width, d.texture.height, (unsigned long long)src.handle);
        }
    }
    if (g_cfg_whole_scene)
        snapshot(cmd_list, dsv, 0, false, true);
    return false;   // never suppress the game's own clear
}

static bool on_draw_indirect(command_list *cmd_list, indirect_command type, resource, uint64_t, uint32_t draw_count, uint32_t)
{
    // dispatches are compute, not geometry - they must not count toward the filter
    if (!g_cfg_count_draws) return false;
    if (type != indirect_command::draw && type != indirect_command::draw_indexed)
        return false;
    cmd_state &cs = CmdState(cmd_list);
    cs.draws += (draw_count ? draw_count : 1);
    note_draw_viewport(cs);
    return false;
}

static void on_init_swapchain(swapchain *sc, bool)
{
    device *const dev = sc->get_device();
    std::lock_guard<std::mutex> lock(g_mutex);
    device_state &ds = g_devices[dev];
    ds.back_buffers.clear();
    for (uint32_t i = 0, n = sc->get_back_buffer_count(); i < n; ++i)
    {
        const resource bb = sc->get_back_buffer(i);
        ds.back_buffers.insert(bb.handle);
        const resource_desc d = dev->get_resource_desc(bb);
        ds.bb_w = d.texture.width; ds.bb_h = d.texture.height;
    }
    ds.display_rect_set = false;
    ds.logged_rect = -1;
    // The swapchain's window is often a child widget whose title is the Qt class name
    // ("duckstation-qt"), not the game - the game name lives on the top-level window that
    // owns it, so resolve that here and keep it.
    {
        const HWND surface = (HWND)sc->get_hwnd();
        const HWND root    = (surface != nullptr) ? GetAncestor(surface, GA_ROOT) : nullptr;
        ds.root_hwnd = (root != nullptr) ? root : surface;
    }
    Log("[depth-charge] swapchain %ux%u, %zu back buffer(s)", ds.bb_w, ds.bb_h, ds.back_buffers.size());
}

static void on_present(command_queue *, swapchain *sc, const rect *, const rect *, uint32_t, const rect *)
{
    device *const dev = sc->get_device();
    std::lock_guard<std::mutex> lock(g_mutex);
    device_state &ds = g_devices[dev];
    if ((++ds.frames % 600) == 0)
    {
        if (ds.snapshots == 0)
            Log("[depth-charge] frame %llu: NO depth pass captured (need %u draws with depth bound). "
                "Seen so far: %llu target binds (%llu with depth), %llu render passes (%llu with depth), %llu draws.",
                (unsigned long long)ds.frames, g_min_draws,
                (unsigned long long)ds.rt_binds, (unsigned long long)ds.dsv_binds,
                (unsigned long long)ds.rp_begins, (unsigned long long)ds.rp_depth,
                (unsigned long long)g_draw_total.load());
        else
            Log("[depth-charge] frame %llu: %llu snapshots (%llu under the %u-draw filter, sparse_refresh %s), last from a %u-draw pass | back-buffer draws seen: %u%s",
                (unsigned long long)ds.frames, (unsigned long long)ds.snapshots, (unsigned long long)ds.sparse_snaps, g_min_draws,
                g_cfg_sparse_refresh ? "on" : "off", ds.last_draws,
                ds.bb_binds, ds.display_rect_set ? "" : " (NEVER matched a back buffer)");
    }
    if (g_survey_frames > 0 && --g_survey_frames == 0)
        Log("[depth-charge] Survey end: back buffer %ux%u, %u passes merged, chosen pass had %u draws",
            ds.bb_w, ds.bb_h, ds.merged, ds.best_draws);

    // The rectangle the emulator draws its picture into, and the shape of the depth we
    // captured. When those disagree the overlay lands in the wrong place, and there is
    // no way to tell which of the two is wrong without seeing both.
    if (g_survey_frames > 0)
        Log("[depth-charge]   display rect: %s  (%d,%d)-(%d,%d) = %ux%u   depth %ux%u   back buffer %ux%u",
            ds.display_rect_set ? "set" : "NOT SET",
            ds.display_rect.left, ds.display_rect.top, ds.display_rect.right, ds.display_rect.bottom,
            ds.display_rect_set ? ds.display_rect.width() : 0,
            ds.display_rect_set ? ds.display_rect.height() : 0,
            ds.raw_w, ds.raw_h, ds.bb_w, ds.bb_h);

    // Commit the frame's largest render viewport as the crop, but only once the same
    // shape has held for a few frames in a row. A single odd frame - a cut scene, a
    // pause, a pass that lost its race - must not move the picture.
    if ((uint64_t)ds.vp_frame_w * ds.vp_frame_h > (uint64_t)ds.vp_win_w * ds.vp_win_h)
    {
        ds.vp_win_w = ds.vp_frame_w; ds.vp_win_h = ds.vp_frame_h;
        ds.vp_win_x = ds.vp_frame_x; ds.vp_win_y = ds.vp_frame_y;
    }
    ds.vp_frame_w = 0; ds.vp_frame_h = 0; ds.vp_frame_x = 0; ds.vp_frame_y = 0;

    // The first measurement lands immediately so the picture is never wrong for two
    // seconds at startup; after that the crop only moves on a window boundary, and
    // only to the largest viewport that window saw.
    const uint32_t kVpWindow = 120;
    if (++ds.vp_win_frames >= kVpWindow || (ds.raw_vp_h == 0 && ds.vp_win_h != 0))
    {
        if (ds.vp_win_w != 0 && ds.vp_win_h != 0 &&
            (ds.vp_win_w != ds.raw_vp_w || ds.vp_win_h != ds.raw_vp_h ||
             ds.vp_win_x != ds.raw_vp_x || ds.vp_win_y != ds.raw_vp_y))
        {
            Log("[depth-charge] picture area %ux%u at %u,%u inside a %ux%u capture%s",
                ds.vp_win_w, ds.vp_win_h, ds.vp_win_x, ds.vp_win_y, ds.raw_w, ds.raw_h,
                (ds.vp_win_w == ds.raw_w && ds.vp_win_h == ds.raw_h) ? " (fills it)" : "  <- cropping to this");
            ds.raw_vp_w = ds.vp_win_w; ds.raw_vp_h = ds.vp_win_h;
            ds.raw_vp_x = ds.vp_win_x; ds.raw_vp_y = ds.vp_win_y;
        }
        ds.vp_win_w = 0; ds.vp_win_h = 0; ds.vp_win_x = 0; ds.vp_win_y = 0; ds.vp_win_frames = 0;
    }

    // The calibration deadline has to be checked somewhere that runs every frame.
    // It used to live in the linearisation path, which only runs once depth has been
    // captured - so on an emulator that publishes no depth at all, Depth Finder sat
    // at the end of its progress bar forever instead of reporting the failure.
    {
        const double now = (double)GetTickCount64() / 1000.0;
        if (g_cal_state == CAL_CAPTURING && now - g_cal_started >= g_cal_seconds + 0.5)
            CalibrateFinish();
        // ...and the same for leaving the result state. Both transitions used to sit
        // in the depth path, so with no depth the run could neither finish nor clear,
        // which left the panel showing a result and no button to try again.
        else if (g_cal_state == CAL_SHOWING && now - g_cal_started >= 3.0)
        {
            g_cal_state = CAL_IDLE;
            g_dbg_view  = g_dbg_view_before;
        }
    }

    // The draw filter assumes an emulator issues many draws per scene pass. That is
    // true of PS2 and GameCube, but a PS1 frame batched by DuckStation can be a
    // handful of draws in total, so a fixed floor of 20 rejects everything. Rather
    // than expose yet another knob, relax it when nothing has EVER been captured
    // the busiest-pass rule still picks the right target once something qualifies.
    if (ds.snapshots == 0 && g_min_draws > 1 && ds.frames >= 300 && (ds.frames % 300) == 0)
    {
        g_min_draws = (g_min_draws > 3) ? (g_min_draws / 3) : 1;
        Log("[depth-charge] no depth pass has cleared the draw filter in %llu frames - "
            "lowering it to %u draws", (unsigned long long)ds.frames, g_min_draws);
    }

    // Decay the incumbent's high-water mark slowly, so a scene that legitimately grows a
    // different pass can take over instead of the first winner holding the title forever.
    if ((ds.frames % 600) == 0 && ds.win_draws > 1) ds.win_draws = (ds.win_draws * 9) / 10;
    if (ds.ready_grace > 0) --ds.ready_grace;
    if (g_cfg_whole_scene)
    {
        // Nothing to capture means this game never clears the depth target we are
        // watching, so whole-scene capture has no boundary to work from. Say so
        // rather than publishing a frozen picture and letting it look like a bug.
        if (ds.iv_best == 0 && ds.frames > 240 && (ds.frames % 600) == 0)
            Log("[depth-charge] whole-scene capture is on but no depth clear was seen this "
                "frame - this game does not clear between scenes, so turn it off");
        ds.iv_best = 0;
    }
    ds.best_draws = 0;
    ds.best_ready = false;
    ds.frame_target = 0;
    ds.last_merged = ds.merged;
    ds.merged = 0;
    if (g_cfg_poll_config && (ds.frames % 180) == 0) ReadConfig(false);
    // The title has to be re-checked periodically, not just once: DuckStation renames its
    // window per game, so the profile key changes while the swapchain stays the same.
    //
    // But GetWindowTextA on a window owned by ANOTHER THREAD does not read a variable - it
    // sends WM_GETTEXT and blocks until that thread pumps messages. on_present arrives on
    // several threads here, and during a level load the UI thread is busy and not pumping,
    // so this hung the renderer every 120 frames and took BioShock Remastered down on every
    // save load. SendMessageTimeout with SMTO_ABORTIFHUNG is the documented way to ask a
    // window for its text without betting on it answering. The HWND itself is now resolved
    // once at swapchain creation instead of being pulled off the swapchain each time.
    // Re-checking the title is now opt-in (poll_title, default off) plus a button in the
    // overlay. Only emulators rename their window mid-session, and even then only when the
    // user loads a different game - polling twice a second to catch that is a bad trade for
    // a call that can block on another thread's message pump.
    if (g_cfg_hud_split)
    {
        ds.hud_index = (ds.hud_next >= 0) ? ds.hud_next : ds.hud_index;
        if (ds.hud_index < 0 && (ds.frames % 300) == 0)
            Log("[depth-charge] HUD split: %d full-screen colour passes with no depth this "
                "frame, busiest drew %u (need %u). back buffer %ux%u",
                ds.co_index, ds.hud_seen_best, g_cfg_hud_min_draws, ds.bb_w, ds.bb_h);
        ds.co_seen   = ds.co_index;
        ds.hud_next  = -1;
        ds.hud_best  = 0;
        ds.hud_seen_best = 0;
        ds.co_index  = 0;
        ds.hud_done  = false;
        ds.hud_inside = false;
    }

    // One read, once, when the game has settled. Never again unless the button is used.
    const bool title_once = (ds.frames == kTitleOnceFrame);
    if ((g_cfg_poll_title || g_title_recheck || title_once) && ds.root_hwnd != nullptr &&
        (g_title_recheck || title_once || (ds.frames % 120) == 0))
    {
        char title[256] = {};
        DWORD_PTR res = 0;
        if (SendMessageTimeoutA(ds.root_hwnd, WM_GETTEXT, (WPARAM)sizeof(title), (LPARAM)title,
                                SMTO_ABORTIFHUNG | SMTO_BLOCK, 50, &res) != 0 && title[0] != '\0')
            SelectGame(title);
        g_title_recheck = false;
    }
}

static void on_destroy_device(device *dev)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_devices.find(dev);
    if (it == g_devices.end()) return;
    device_state &ds = it->second;
    if (ds.cal_rtv    != 0) dev->destroy_resource_view(ds.cal_rtv);
    if (ds.cal_rt     != 0) dev->destroy_resource(ds.cal_rt);
    if (ds.cal_stage  != 0) dev->destroy_resource(ds.cal_stage);
    if (ds.cal_pso    != 0) dev->destroy_pipeline(ds.cal_pso);
    if (ds.cal_layout != 0) dev->destroy_pipeline_layout(ds.cal_layout);
    if (ds.bar_copy_srv != 0) dev->destroy_resource_view(ds.bar_copy_srv);
    if (ds.bar_copy     != 0) dev->destroy_resource(ds.bar_copy);
    if (ds.bar_rtv      != 0) dev->destroy_resource_view(ds.bar_rtv);
    if (ds.bar_rt       != 0) dev->destroy_resource(ds.bar_rt);
    if (ds.bar_stage    != 0) dev->destroy_resource(ds.bar_stage);
    if (ds.bar_pso      != 0) dev->destroy_pipeline(ds.bar_pso);
    if (ds.bar_layout   != 0) dev->destroy_pipeline_layout(ds.bar_layout);
    if (ds.dbg_pso    != 0) dev->destroy_pipeline(ds.dbg_pso);
    if (ds.dbg_layout != 0) dev->destroy_pipeline_layout(ds.dbg_layout);
    if (ds.lin_rtv != 0) dev->destroy_resource_view(ds.lin_rtv);
    if (ds.lin_srv != 0) dev->destroy_resource_view(ds.lin_srv);
    if (ds.lin     != 0) dev->destroy_resource(ds.lin);
    if (ds.convert_sampler != 0) dev->destroy_sampler(ds.convert_sampler);
    if (ds.convert_pso     != 0) dev->destroy_pipeline(ds.convert_pso);
    if (ds.convert_layout  != 0) dev->destroy_pipeline_layout(ds.convert_layout);
    if (ds.disp_srv != 0) dev->destroy_resource_view(ds.disp_srv);
    if (ds.disp != 0 && ds.disp != ds.raw) dev->destroy_resource(ds.disp);
    if (ds.raw != 0) dev->destroy_resource(ds.raw);
    g_devices.erase(it);
}

extern "C" __declspec(dllexport) const char *NAME        = "Greenskull's Depth Charge";
extern "C" __declspec(dllexport) const char *AUTHOR      = "Greenskull";
extern "C" __declspec(dllexport) const char *WEBSITE     = "https://github.com/GreenskullAI/depth-charge";
extern "C" __declspec(dllexport) const char *DESCRIPTION =
    "Finds the game's depth buffer, measures its curve and places it under the picture, "
    "then publishes it as ordinary ReShade depth. Built for emulators, where depth is often "
    "somewhere Generic Depth does not look, or does not line up with the picture. "
    "Disable ReShade's Generic Depth when using this.";

static bool s_reg_cmd_lifetime = false;   // whether DllMain registered the command-list lifetime events

BOOL APIENTRY DllMain(HMODULE module, DWORD reason, LPVOID)
{
    switch (reason)
    {
    case DLL_PROCESS_ATTACH:
        if (!reshade::register_addon(module)) return FALSE;
        if (GetModuleFileNameA(module, g_cfg_path, MAX_PATH) != 0)
        {
            // Always depth-charge.cfg beside the add-on, whatever the add-on file is called. It used
            // to be named after the add-on, so a browser saving a second download as
            // "depth-charge (1).addon64" quietly started the player on empty settings (Oct 20 2026).
            char *slash = strrchr(g_cfg_path, '\\');
            if (char *fwd = strrchr(g_cfg_path, '/'); fwd != nullptr && (slash == nullptr || fwd > slash))
                slash = fwd;
            char *name = slash != nullptr ? slash + 1 : g_cfg_path;
            strcpy_s(name, sizeof(g_cfg_path) - (name - g_cfg_path), "depth-charge.cfg");
            ReadConfig(true);
        }
        if (g_cfg_hook_swapchain)
            reshade::register_event<reshade::addon_event::init_swapchain>(on_init_swapchain);
        // ReShade only wraps and tracks command lists / deferred contexts when an add-on
        // actually subscribes to these events, so REGISTERING them changes the host's own
        // behaviour regardless of what the callbacks do. The cfg is read above, before this
        // point, so these gates can suppress registration entirely - which is the only way
        // to tell "our logic is at fault" apart from "subscribing at all is at fault".
        if (g_cfg_track_binds)
        {
            reshade::register_event<reshade::addon_event::bind_viewports>(on_bind_viewports);
            reshade::register_event<reshade::addon_event::bind_scissor_rects>(on_bind_scissors);
            reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_bind_rtv_dsv);
        }
        // Remembered rather than re-read at unload: the cfg is re-read while the game runs, and a
        // switch flipped mid-session must not leave a callback registered into an unloaded DLL.
        s_reg_cmd_lifetime = g_cfg_track_cmd_lifetime && (g_cfg_track_binds || g_cfg_track_passes || g_cfg_count_draws);
        if (s_reg_cmd_lifetime)
        {
            reshade::register_event<reshade::addon_event::init_command_list>(on_init_command_list);
            reshade::register_event<reshade::addon_event::destroy_command_list>(on_destroy_command_list);
        }
        if (g_cfg_track_passes)
        {
            reshade::register_event<reshade::addon_event::begin_render_pass>(on_begin_render_pass);
            reshade::register_event<reshade::addon_event::end_render_pass>(on_end_render_pass);
        }
        if (g_cfg_whole_scene || g_cfg_track_binds)
            reshade::register_event<reshade::addon_event::clear_depth_stencil_view>(on_clear_dsv);
        if (g_cfg_count_draws)
        {
            reshade::register_event<reshade::addon_event::draw>(on_draw);
            reshade::register_event<reshade::addon_event::draw_indexed>(on_draw_indexed);
            reshade::register_event<reshade::addon_event::draw_or_dispatch_indirect>(on_draw_indirect);
        }
        if (g_cfg_hook_effects)
        {
            reshade::register_event<reshade::addon_event::reshade_begin_effects>(on_begin_effects);
            reshade::register_event<reshade::addon_event::reshade_reloaded_effects>(on_reloaded_effects);
            reshade::register_event<reshade::addon_event::reshade_finish_effects>(on_finish_effects);
        }
        if (g_cfg_hook_present)
            reshade::register_event<reshade::addon_event::present>(on_present);
        if (g_cfg_hook_swapchain)
            reshade::register_event<reshade::addon_event::destroy_device>(on_destroy_device);
        if (g_cfg_hook_overlay)
            reshade::register_overlay(nullptr, draw_settings);
        Log("[depth-charge] loaded - registered: binds=%s passes=%s draws=%s lifetime=%s present=%s effects=%s swapchain=%s overlay=%s",
            g_cfg_track_binds ? "y" : "N", g_cfg_track_passes ? "y" : "N", g_cfg_count_draws ? "y" : "N",
            s_reg_cmd_lifetime ? "y" : "N",
            g_cfg_hook_present ? "y" : "N", g_cfg_hook_effects ? "y" : "N", g_cfg_hook_swapchain ? "y" : "N", g_cfg_hook_overlay ? "y" : "N");
        Log("[depth-charge]   present sub-gates: poll_config=%s poll_title=%s",
            g_cfg_poll_config ? "y" : "N", g_cfg_poll_title ? "y" : "N");
        break;
    case DLL_PROCESS_DETACH:
        // ReShadeRegisterEvent stores a BARE FUNCTION POINTER with no module identity, so
        // every event registered above must be handed back explicitly. The overlay was being
        // unregistered here but the fourteen events never were - and ReShade loads and
        // unloads add-ons repeatedly during startup (BioShock Remastered: three full
        // load/unload cycles before the one that sticks). Each unload takes this DLL's code
        // pages with it while those callback addresses may still be live, so the next call
        // into one lands in unmapped memory. Mirrors the registration, gates included.
        if (g_cfg_track_binds)
        {
            reshade::unregister_event<reshade::addon_event::bind_viewports>(on_bind_viewports);
            reshade::unregister_event<reshade::addon_event::bind_scissor_rects>(on_bind_scissors);
            reshade::unregister_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_bind_rtv_dsv);
        }
        if (s_reg_cmd_lifetime)
        {
            reshade::unregister_event<reshade::addon_event::init_command_list>(on_init_command_list);
            reshade::unregister_event<reshade::addon_event::destroy_command_list>(on_destroy_command_list);
        }
        if (g_cfg_track_passes)
        {
            reshade::unregister_event<reshade::addon_event::begin_render_pass>(on_begin_render_pass);
            reshade::unregister_event<reshade::addon_event::end_render_pass>(on_end_render_pass);
        }
        if (g_cfg_whole_scene || g_cfg_track_binds)
            reshade::unregister_event<reshade::addon_event::clear_depth_stencil_view>(on_clear_dsv);
        if (g_cfg_count_draws)
        {
            reshade::unregister_event<reshade::addon_event::draw>(on_draw);
            reshade::unregister_event<reshade::addon_event::draw_indexed>(on_draw_indexed);
            reshade::unregister_event<reshade::addon_event::draw_or_dispatch_indirect>(on_draw_indirect);
        }
        if (g_cfg_hook_swapchain)
        {
            reshade::unregister_event<reshade::addon_event::init_swapchain>(on_init_swapchain);
            reshade::unregister_event<reshade::addon_event::destroy_device>(on_destroy_device);
        }
        if (g_cfg_hook_effects)
        {
            reshade::unregister_event<reshade::addon_event::reshade_begin_effects>(on_begin_effects);
            reshade::unregister_event<reshade::addon_event::reshade_reloaded_effects>(on_reloaded_effects);
            reshade::unregister_event<reshade::addon_event::reshade_finish_effects>(on_finish_effects);
        }
        if (g_cfg_hook_present)
            reshade::unregister_event<reshade::addon_event::present>(on_present);
        if (g_cfg_hook_overlay)
            reshade::unregister_overlay(nullptr, draw_settings);
        reshade::unregister_addon(module);
        break;
    }
    return TRUE;
}

