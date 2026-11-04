# Greenskull's Depth Charge

A ReShade add-on that finds a game's depth buffer, measures it and places it under the picture, so effects that need depth (ambient occlusion, reflections, DLSS 5 feeds) work in emulators.

ReShade's own "Generic Depth" add-on picks a depth buffer and leaves the rest to you. Emulators are where that tends to fall apart. Depth Charge handles three things for you:

- **When to capture.** It copies depth the moment the game finishes drawing into it, before the emulator can reuse the buffer. It also finds depth kept somewhere unusual, such as DuckStation's PS1 depth.
- **Where it goes.** Emulators fit a 4:3 or 16:9 picture into the window between black bars. Depth Charge places depth over exactly that picture, at any internal resolution and window shape.
- **What it means.** Press one button, hold still for a few seconds, and Depth Charge measures the depth and converts it to real distance. It then writes it in the format ReShade's own depth settings expect, so you leave those settings alone.

The result is ordinary ReShade depth, so any effect that reads depth should work with no changes. We test with [LumeniteFX](https://github.com/umar-afzaal/LumeniteFX).

***

## Tested games

**PCSX2 (PlayStation 2):** Crazy Frog Racer 2, Final Fantasy X, Front Mission 4, Grand Theft Auto: San Andreas, Guitar Hero II, Kingdom Hearts, Metal Gear Solid 3: Snake Eater, SoulCalibur III, Spider-Man 2, Star Wars: Battlefront II, Tekken Tag Tournament, TimeSplitters 2, Tony Hawk's Underground 2, WWE SmackDown! vs. RAW

**DuckStation (PlayStation 1):** Brave Fencer Musashi, Crash Bandicoot, MediEvil, Silent Hill, Spyro the Dragon, The Legend of Dragoon, Tomb Raider II

**Dolphin (GameCube):** Mario Kart: Double Dash!!, Skies of Arcadia: Legends

**PC:** BioShock Remastered [it needs a few extra steps, see "BioShock Remastered" below]

These games come with built-in settings, so you can skip the Depth Finder step. Depth Charge recognises them by the name in the window title, so another region or version of a game may still need one press of **Depth Finder**, like any other game (see "Set up a game").

***

## What you need

- Windows.
- ReShade **with full add-on support**, from [reshade.me](https://reshade.me). [The standard version can't load add-ons.]
- A game or emulator running Direct3D 10, 11 or 12. [Step 1 of "Install Depth Charge" shows where to set this in each emulator.]
- Effects that use depth. Depth Charge has no effects of its own. We use [LumeniteFX](https://github.com/umar-afzaal/LumeniteFX) by Umar Afzaal, which ReShade's installer can download for you.

***

## Install ReShade

1. Download ReShade **with full add-on support** from [reshade.me](https://reshade.me) and run it.

2. Click **Browse** and select the emulator's own `.exe` [for example `pcsx2-qt.exe`, `duckstation-qt-x64-ReleaseLTCG.exe` or `Dolphin.exe`]. For a PC game, select the game's `.exe`.

3. When it asks which rendering API to use, click **DirectX 10/11/12**. [One button covers all three.]

4. On the effects screen, pick the effects you want. Leave **Standard effects** and **SweetFX** ticked [they start ticked]. Any effects that use depth should work with Depth Charge. The ones we use and test with are in **LumeniteFX**: scroll down and tick it, and more options appear under it. Tick **Kernel**, **SSSR** and **LSAO** [the Kernel prepares the scene for the other two, which are reflections and ambient occlusion].

5. On the add-ons screen, you don't need to tick anything. [Depth Charge isn't on that list. You add it by hand below.]

6. Let the installer finish.

***

## Install Depth Charge

1. Set the emulator to Direct3D 11 or 12. [Depth Charge needs one of these to measure and convert depth.]
    - **PCSX2:** Settings > Graphics, **Graphics API**: Direct3D 11 or Direct3D 12.
    - **DuckStation:** Settings > Graphics, **Renderer**: Direct3D 11 or Direct3D 12.
    - **Dolphin:** Options > Graphics Settings, **Backend**: Direct3D 11 or Direct3D 12.
    - **A PC game** has to run on DirectX 10, 11 or 12 itself.

2. Download `depth-charge.addon64` from the [Releases page](../../releases/latest). Copy it into the same folder as the game or emulator's `.exe` [the one you picked in ReShade's installer]. For a 32-bit game, use `depth-charge.addon32` instead. [PCSX2, DuckStation and Dolphin are 64-bit. For a PC game you're not sure about, try the 64-bit one first. If Depth Charge doesn't appear in ReShade's Add-ons tab, swap it for the 32-bit one.]

3. Start the game or emulator, press **Home** to open ReShade, go to the **Add-ons** tab and untick **Generic Depth**. Then restart the game or emulator. [Both try to supply depth, and whichever runs last wins.]

4. **DuckStation (PS1) only.** [DuckStation only makes a depth buffer when one of its own shaders asks for one. `DepthCharge_EnableDepth.fx` is that ask, and it doesn't change the picture.]
    1. Copy `DepthCharge_EnableDepth.fx` into DuckStation's `resources\shaders\reshade\Shaders` folder.
    2. In DuckStation, go to Settings > Post-Processing, then the **Internal** tab [not Display]. Turn on post-processing and add **DepthCharge_EnableDepth**.
    3. In Settings > Graphics > PGXP, turn on **Geometry Correction** and **Depth Buffer**.
    4. Restart the game. [DuckStation only picks up internal shader changes after a restart.]

You don't need to change ReShade's depth settings (the `RESHADE_DEPTH_...` preprocessor definitions). Leave them as they are. [Depth Charge reads them and writes depth to match.]

***

## Set up a game

1. Start the game and get into gameplay [not a menu or a cutscene].

2. Press **Home** to open ReShade, go to the **Add-ons** tab and open **Greenskull's Depth Charge**.

3. Look at the line with the game's name. If it says **tested defaults**, this game is already set up: skip to step 6.

4. Leave **Picture aspect** on **Auto**. It finds the picture's shape from the black bars around it. If you later change the emulator's own aspect setting, press **Check again** [next to the Auto message]. If effects still look shifted, pick the shape by hand [usually 16:9 or 4:3].

5. Look toward a long view [down a corridor or toward the horizon], press **Depth Finder**, and hold still for a few seconds. [A debug view shows depth as a gradient from blue (close) to red (far). That means it worked.] If it didn't work, the panel says why [for example "Could not separate near from far"] and the colours stay flat or patchy instead of running smoothly from near to far. Try again somewhere else, with more of the scene in view. If it still won't measure, the game may store its depth in a way Depth Charge doesn't handle yet.

6. Turn on the effects you want in the **Home** tab. For LumeniteFX, tick **LUMENITE: Kernel 2.0**, **LUMENITE: SSSR** and **LUMENITE: LSAO**, and keep the Kernel above the other two in the list [drag it up if it isn't; the other two use what the Kernel builds].

7. That's it. Depth Charge remembers the result for this game in `depth-charge.cfg`, next to the add-on. [Copy that file to keep your measurements or share them.]

***

## BioShock Remastered

BioShock Remastered works the same way, with four differences:

1. In ReShade's installer, select `BioshockHD.exe`. It's in the `Build\Final` folder inside the game's Steam folder [Steam: right-click the game, Manage, Browse local files].

2. BioShock Remastered is 32-bit, so copy `depth-charge.addon32` into that `Build\Final` folder, not the 64-bit one.

3. In Depth Charge's panel, open **Advanced** and tick **Use the whole scene, not the busiest pass**. [BioShock clears its depth and draws your weapon on top. Without this, Depth Charge can pick up the weapon instead of the level.]

4. If you save a ReShade preset, don't name it `Bioshock.ini`. [That's the name of the game's own settings file in the same folder, and the game won't start. ReShade's default name, `ReShadePreset.ini`, is fine.]

It comes with built-in settings, so you can skip Depth Finder.

***

## DLSS 5

Depth Charge was first written to feed [DLSS5-Feeder](https://github.com/jlrouzies-fr/DLSS5-Feeder). Install DLSS5-Feeder by its own guide, with one difference: its guide turns **Generic Depth** on. Leave it off. [Depth Charge supplies the depth instead.]

Tested with DLSS5-Feeder 0.12.0.

***

## Online games

ReShade's add-on version can trigger anti-cheat in online games, which can lead to a ban. **By installing Depth Charge, you agree not to use it in competitive online games.**

***

## If something looks wrong

- **Effects sit in the wrong place, or stop short of the edge of the picture.** If Picture aspect is on Auto, press **Check again**. If that doesn't fix it, set **Picture aspect** by hand to match what the game shows.
- **No depth at all.** Check that Generic Depth is off (Install Depth Charge, step 3) and the emulator is on Direct3D 11 or 12 (step 1). For PS1, check step 4.
- **You switched games and depth looks wrong.** Press **Check game**. [It reads the window title and loads that game's settings.]
- **Depth looks flat, or near and far look the same.** Press **Depth Finder** again during gameplay, looking toward a long view.

***

## Credits

- [crosire](https://github.com/crosire/reshade): ReShade and its add-on API
- [jlrouzies-fr](https://github.com/jlrouzies-fr/DLSS5-Feeder): DLSS5-Feeder, the project Depth Charge was first written to feed
- [Umar Afzaal](https://github.com/umar-afzaal/LumeniteFX): LumeniteFX, the effects we test with
- [stenzek](https://github.com/stenzek/duckstation): DuckStation, and its PGXP depth buffer
- iCatButler: the first PS1 depth buffer built from PGXP
- Built with Claude (Anthropic)

***

## License

MIT, copyright Dan Hammill (Greenskull). See [LICENSE](LICENSE).

Depth Charge contains no LumeniteFX code. LumeniteFX has its own licence and is downloaded from its author's official links. Depth Charge is compiled against ReShade's add-on headers (BSD 3-Clause or MIT) and Dear ImGui's headers (MIT).
