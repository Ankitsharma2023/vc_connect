import streamlit as st
import pandas as pd
from tag_extractor import extract_tags_tool

@st.cache_data(show_spinner=True)
def load_data():
    return pd.read_csv("investor_data.csv")

df = load_data()

st.title("Startup Founder - Investor Matchmaking")

startup_idea = st.text_area("Describe your startup idea", height=150)

if st.button("Find Investors"):
    if not startup_idea.strip():
        st.warning("Please enter your startup idea first!")
    else:
        with st.spinner("Analyzing your startup and finding matching investors..."):
            try:
          
                tags = extract_tags_tool.invoke(startup_idea)

                st.subheader("🔖Extracted Tags")
                st.markdown(f"**Domain:** {', '.join(tags['domain'])}")
                st.markdown(f"**Stage:** {tags['stage']}")
                st.markdown(f"**Region:** {tags['region']}")

                filtered_df = df[
                    df['Investment thesis'].str.contains(tags['domain'][0], case=False, na=False) &
                    df['Stage of investment'].str.contains(tags['stage'], case=False, na=False) &
                    df['Countries of investment'].str.contains(tags['region'], case=False, na=False)
                ]

                if filtered_df.empty:
                    st.warning("No matching investors found. Try adjusting your description.")
                else:
                    st.subheader("💼 Top Matching Investors:")
                    for idx, investor in filtered_df.head(5).iterrows():
                        st.markdown(f"### {investor['Investor name']}")
                        st.markdown(f"**Type:** {investor['Investor type']}")
                        st.markdown(f"**Focus:** {investor['Investment thesis']}")
                        st.markdown(f"**Stage:** {investor['Stage of investment']}")
                        st.markdown(f"**Location:** {investor['Countries of investment']}")
                        st.markdown(f"**Check Size:** ${investor['First cheque minimum']} to ${investor['First cheque maximum']}")
                        st.markdown("---")
            except Exception as e:
                st.error(f"Error extracting tags or searching investors: {e}")
