/*
 * Depth Charge: Enable Depth
 * Part of Greenskull's Depth Charge.  Author: Greenskull
 *
 * WHAT THIS IS FOR
 *
 * The PlayStation 1 had no depth buffer at all; it sorted polygons by hand with
 * ordering tables.  DuckStation can reconstruct one from PGXP, and when its own
 * post-processing chain asks for depth it extracts that buffer, once per frame, into
 * a display-sized texture: already cropped to the picture, already the right size.
 * That texture is what Depth Charge captures and republishes to ReShade.
 *
 * But DuckStation only builds it if something in the INTERNAL chain declares a
 * texture with the DEPTH semantic.  Until now the only way to ask was to load a
 * debug view such as DisplayDepth, which answers the question by painting false
 * colour over your game.
 *
 * This shader asks the same question and then does nothing.  It declares the depth
 * texture, samples one texel of it so no compiler can optimise the declaration away,
 * and returns the frame exactly as it arrived.
 *
 * HOW TO USE IT
 *
 *   DuckStation -> Settings -> Post-Processing -> INTERNAL tab (not Display)
 *     Enable Post Processing, then Add -> DepthCharge_EnableDepth
 *
 *   Also required, or there is no depth to extract:
 *     Graphics -> Renderer:  Direct3D 11 or Direct3D 12  (not Vulkan or OpenGL)
 *     Graphics -> PGXP:      Geometry Correction ON, Depth Buffer ON
 *
 * Nothing to configure.  If the picture changes in any way, that is a bug.
 */

#include "ReShade.fxh"

float3 PS_DepthChargeEnableDepth(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
	const float3 colour = tex2D(ReShade::BackBuffer, texcoord).rgb;

	// The depth read has to survive the optimiser, or the DEPTH texture is stripped
	// from the compiled effect and DuckStation stops extracting it. Multiplying by
	// exactly zero keeps the sample in the shader and out of the picture.
	const float keep = tex2D(ReShade::DepthBuffer, texcoord).r * 0.0;

	return colour + keep;
}

technique DepthCharge_EnableDepth
<
	ui_label = "Depth Charge: Enable Depth";
	ui_tooltip = "Makes DuckStation publish its PGXP depth buffer, without changing the picture.\n"
	             "Needs Direct3D 11/12 and PGXP Depth Buffer enabled.\n"
	             "Add this to the INTERNAL post-processing chain, not Display.";
>
{
	pass
	{
		VertexShader = PostProcessVS;
		PixelShader = PS_DepthChargeEnableDepth;
	}
}
