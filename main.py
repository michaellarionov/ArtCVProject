# =============================================================================
# STREAMLIT ART GENERATOR - MAIN APPLICATION
# =============================================================================
# Run with: streamlit run main.py

import streamlit as st
from config import ART_PERIODS, COLOR_PALETTES


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Art Generator",
    page_icon="🎨",
    layout="wide"
)


# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.title("🎨 AI Art Period Generator")
st.markdown("Generate artwork in the style of different historical art periods")


# -----------------------------------------------------------------------------
# SIDEBAR - USER CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Generation Settings")
    
    # Art Period Selection
    selected_period = st.selectbox(
        "Select Art Period",
        options=list(ART_PERIODS.keys())
    )
    
    # Show artists for selected period
    period_info = ART_PERIODS[selected_period]
    st.caption(f"Example artists: {', '.join(period_info['artists'])}")
    
    st.divider()
    
    # Color Palette Selection
    selected_palette = st.selectbox(
        "Select Color Palette",
        options=list(COLOR_PALETTES.keys())
    )
    
    # Display palette colors visually
    palette_info = COLOR_PALETTES[selected_palette]
    cols = st.columns(5)
    for i, color in enumerate(palette_info["colors"]):
        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        with cols[i]:
            st.color_picker(
                label=f"C{i+1}",
                value=hex_color,
                disabled=True,
                key=f"color_{i}"
            )
    
    st.divider()
    
    # Subject Input
    subject = st.text_input(
        "Subject / Scene",
        value="a serene landscape with rolling hills",
        help="Describe what you want in the painting"
    )
    
    # Advanced Settings (collapsed by default)
    with st.expander("Advanced Settings"):
        num_steps = st.slider("Inference Steps", 20, 100, 50)
        guidance = st.slider("Guidance Scale", 1.0, 20.0, 7.5)
    
    st.divider()
    
    # Generate Button
    generate_btn = st.button(
        "🎨 Generate Art",
        type="primary",
        use_container_width=True
    )


# -----------------------------------------------------------------------------
# HELPER FUNCTION - BUILD PROMPT
# -----------------------------------------------------------------------------
def build_prompt(subject: str, period: str, palette: str) -> str:
    """Combine user selections into a generation prompt."""
    period_style = ART_PERIODS[period]["prompt_style"]
    palette_desc = COLOR_PALETTES[palette]["description"]
    
    prompt = f"{subject}, {period_style}, {palette_desc}, masterpiece, highly detailed"
    return prompt


# -----------------------------------------------------------------------------
# MAIN CONTENT AREA
# -----------------------------------------------------------------------------
if generate_btn:
    # Build the prompt from selections
    prompt = build_prompt(subject, selected_period, selected_palette)
    
    # Show the generated prompt
    with st.expander("View Generated Prompt", expanded=False):
        st.code(prompt)
    
    # Generate the image
    with st.spinner("🎨 Creating your artwork... (this may take a minute)"):
        try:
            from inference.generate import generate_art
            
            image = generate_art(
                prompt=prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance
            )
            
            # Display the result
            st.image(image, caption=f"{selected_period} - {selected_palette}")
            
            # Download button
            from io import BytesIO
            buf = BytesIO()
            image.save(buf, format="PNG")
            st.download_button(
                label="📥 Download Image",
                data=buf.getvalue(),
                file_name="generated_art.png",
                mime="image/png"
            )
            
        except Exception as e:
            st.error(f"Generation failed: {str(e)}")

else:
    # Default state
    st.info("👈 Configure your settings in the sidebar and click **Generate Art**")


# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.divider()
st.caption("Powered by Stable Diffusion | Using WikiArt dataset styles")