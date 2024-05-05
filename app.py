import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sec_api import QueryApi, XbrlApi
import dspy
import csv
import json
import os


# Function to standardize filing URLs
def standardize_filing_url(url):
    return url.replace("ix?doc=/", "")  # Remove unnecessary parts from URL


# Function to download 10-K metadata
def get_10K_metadata(ticker="AAPL", start_year=1995, end_year=2023):
    frames = []

    for year in range(start_year, end_year + 1):
        number_of_objects_downloaded = 0

        for month in range(1, 13):
            padded_month = str(month).zfill(2)
            tickername = f"ticker:({ticker})"
            date_range_filter = (
                f"filedAt:[{year}-{padded_month}-01 TO {year}-{padded_month}-31]"
            )
            form_type_filter = 'formType:"10-K" AND NOT formType:("10-K/A", NT)'
            query = (
                tickername + " AND " + date_range_filter + " AND " + form_type_filter
            )

            query_from = 0
            query_size = 200

            while True:
                query = {
                    "query": query,
                    "from": query_from,
                    "size": query_size,
                    "sort": [{"filedAt": {"order": "desc"}}],
                }

                response = queryApi.get_filings(query)
                filings = response["filings"]

                if len(filings) == 0:
                    break
                else:
                    query_from += query_size

                metadata = list(
                    map(
                        lambda f: {
                            "ticker": f["ticker"],
                            "cik": f["cik"],
                            "formType": f["formType"],
                            "filedAt": f["filedAt"],
                            "filingUrl": f["linkToFilingDetails"],
                        },
                        filings,
                    )
                )

                df = pd.DataFrame.from_records(metadata)
                df = df[df["ticker"].str.len() > 0]
                df["filingUrl"] = df["filingUrl"].apply(standardize_filing_url)
                frames.append(df)
                number_of_objects_downloaded += len(df)

    result = pd.concat(frames)
    return result


# Extract HTML links from metadata
def extract_links(data):
    data.to_csv("metadata_10K.csv", index=False)
    df = pd.read_csv("metadata_10K.csv")

    html_links = {}

    for index, items in df.iterrows():
        # Extract company ticker and year from the DataFrame
        year = items["filedAt"][
            :4
        ]  # Assuming year is the first three characters of the filedAt column

        # Save the link to the corresponding name
        link = items["filingUrl"]
        html_links[year] = link

    return html_links


# Fetch JSON data
def fetch_json_data_for_all_links(html_links):
    json_data = {}
    for year, link in html_links.items():
        json_data[year] = xbrlApi.xbrl_to_json(
            htm_url=link
        )  # Fetch JSON data using xbrl API
    return json_data


# API keys and initialization
API_KEY = "API_KEY"
queryApi = QueryApi(api_key=API_KEY)
xbrlApi = XbrlApi(API_KEY)


# Function to get all revenues from JSON data
def get_all_revenues(json_data):
    all_revenues = []
    for year, xbrl_json in json_data.items():
        if (
            "StatementsOfIncome" in xbrl_json
            and "RevenueFromContractWithCustomerExcludingAssessedTax"
            in xbrl_json["StatementsOfIncome"]
        ):
            all_revenues.extend(
                xbrl_json["StatementsOfIncome"][
                    "RevenueFromContractWithCustomerExcludingAssessedTax"
                ]
            )
    all_revenues = pd.json_normalize(all_revenues)
    try:
        all_revenues.drop_duplicates(inplace=True)
    except Exception:
        pass

    all_revenues["value"] = all_revenues["value"].astype(int)

    try:
        all_revenues = all_revenues.explode("segment")

        segment_split = all_revenues["segment"].apply(pd.Series)
        segment_split = segment_split.rename(
            columns={"dimension": "segment.dimension", "value": "segment.value"}
        )
        segment_split = segment_split.drop(0, axis=1)

        all_revenues = all_revenues.combine_first(segment_split)
        all_revenues = all_revenues.drop("segment", axis=1)
        all_revenues = all_revenues.reset_index(drop=True)
    except Exception:
        pass
    return all_revenues


# Function to get revenues product wise
def get_revenue_product(all_revenues, company_name):
    segment_labels = all_revenues["segment.value"].dropna().unique().tolist()
    mask = all_revenues["segment.dimension"] == "srt:ProductOrServiceAxis"
    revenue_product = all_revenues[mask]

    try:
        revenue_product = revenue_product.drop_duplicates(
            subset=["period.endDate", "segment.value"]
        )
    except Exception:
        pass
    revenue_product_pivot = revenue_product.pivot(
        index="period.endDate", columns="segment.value", values="value"
    )
    return revenue_product, revenue_product_pivot, segment_labels


# Function to plot revenue by product
def plot_revenue_by_product(
    revenue_product, revenue_product_pivot, company_name, segment_labels
):
    fig, ax = plt.subplots(figsize=(8, 6))
    revenue_product_pivot.plot(kind="bar", stacked=True, ax=ax)
    plt.xticks(rotation=0)
    ax.set_title(
        f"{company_name}'s Revenue by Product Category", fontsize=16, fontweight="bold"
    )
    ax.set_xlabel("Period End Date", fontsize=12)
    ax.set_ylabel("Revenue (USD)", fontsize=12)
    ax.legend(
        title="Product Category", loc="upper left", fontsize="small", title_fontsize=10
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    formatter = ticker.FuncFormatter(lambda x, pos: "$%1.0fB" % (x * 1e-9))
    plt.gca().yaxis.set_major_formatter(formatter)
    label_map = {label: label.split(":")[-1] for label in segment_labels}
    new_labels = [
        label_map[label] for label in sorted(revenue_product["segment.value"].unique())
    ]
    handles, _ = ax.get_legend_handles_labels()
    plt.legend(handles=handles[::-1], labels=new_labels[::-1])
    for p in ax.containers:
        ax.bar_label(
            p,
            labels=["%.1f" % (v / 1e9) for v in p.datavalues],
            label_type="center",
            fontsize=8,
        )
    return fig


# Function to get revenues region wise
def get_revenue_region(all_revenues, company_name):
    segment_labels = all_revenues["segment.value"].dropna().unique().tolist()
    mask = all_revenues["segment.dimension"] == "srt:StatementGeographicalAxis"
    revenue_product = all_revenues[mask]
    try:
        revenue_product = revenue_product.drop_duplicates(
            subset=["period.endDate", "segment.value"]
        )
    except Exception:
        pass
    revenue_product_pivot = revenue_product.pivot(
        index="period.endDate", columns="segment.value", values="value"
    )
    return revenue_product, revenue_product_pivot, segment_labels


# Function to plot revenue by region
def plot_revenue_by_region(
    revenue_product, revenue_product_pivot, company_name, segment_labels
):
    fig, ax = plt.subplots(figsize=(8, 6))
    revenue_product_pivot.plot(kind="bar", stacked=True, ax=ax)
    plt.xticks(rotation=0)
    ax.set_title(
        f"{company_name}'s Revenue by Region Category", fontsize=16, fontweight="bold"
    )
    ax.set_xlabel("Period End Date", fontsize=12)
    ax.set_ylabel("Revenue (USD)", fontsize=12)
    ax.legend(title="Region", loc="upper left", fontsize="small", title_fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    formatter = ticker.FuncFormatter(lambda x, pos: "$%1.0fB" % (x * 1e-9))
    plt.gca().yaxis.set_major_formatter(formatter)
    label_map = {label: label.split(":")[-1] for label in segment_labels}
    new_labels = [
        label_map[label] for label in sorted(revenue_product["segment.value"].unique())
    ]
    handles, _ = ax.get_legend_handles_labels()
    plt.legend(handles=handles[::-1], labels=new_labels[::-1])
    for p in ax.containers:
        ax.bar_label(
            p,
            labels=["%.1f" % (v / 1e9) for v in p.datavalues],
            label_type="center",
            fontsize=8,
        )
    return fig


# Function to read from a json file
def load_json_data(filename):
    with open(filename, "r") as file:
        return json.load(file)


# LLM Model Configuration
def configure_model():
    ollama = dspy.OllamaLocal(
        model="llama2", timeout_s=60, max_tokens=5000
    )  # llama2 chosen
    dspy.settings.configure(lm=ollama)


## LLM Pipeline and Signatures


# Signature to analyse text
class Analyse(dspy.Signature):
    """Analyse the whole text and give valuable and useful insights."""

    question = dspy.InputField(desc="The text to be analysed.")
    answer = dspy.OutputField(
        desc="Generated insight or information derived from the text data."
    )


# Signature to provide summary for the given company
class SummarizeText(dspy.Signature):
    """Summarize details about the company from it's ticker name. Summary should include:
    1. Founding details,
    2. Headquarters
    3. Products
    4. Revenue sources
    5. Competitors
    6. Market capitalization
    7. CEO
    8. Focus areas
    9. Future Plans"""

    question = dspy.InputField(desc="The company ticker.")
    answer = dspy.OutputField(desc="A concise summary of the company.")


class AnalyseBot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.Predict(Analyse)

    def forward(self, question):
        answer = self.generate_answer(question=question)

        return answer.answer


# Module for analysing csv data using Analyse signature
class CSVAnalyseBot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.Predict(Analyse)

    def forward(self, question):
        with open(question, newline="") as csvfile:
            csv_reader = csv.reader(csvfile)
            info_lines = []
            for row in csv_reader:
                info_lines.append(",".join(row))
            info = "\n".join(info_lines)
        answer = self.generate_answer(question=info)
        answer = answer.answer
        try:
            answer = answer.split("Answer:")[1].strip()
        except IndexError:
            pass
        return answer


class SummariseBot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.Predict(SummarizeText)

    def forward(self, question):
        answer = self.generate_answer(question=question)
        answer = answer.answer
        try:
            answer = answer.split("Answer:")[1].strip()
            return answer
        except IndexError:
            return answer.answer


# Initialising different bots
analyse = AnalyseBot()
csv_bot = CSVAnalyseBot()
sum_bot = SummariseBot()


# Streamlit app Initialisation
st.title("Financial Analysis App")


## Main Function
def main():
    configure_model()  # Configure the LLM
    company = st.text_input(
        "Enter company ticker: "
    ).upper()  # User inputs the company ticker
    if company:
        # Download metadata for the company's 10-K filings
        try:
            metadata_10K = get_10K_metadata(ticker=company)
            st.write("Financial reports downloaded successfully.")
            links = extract_links(metadata_10K)
            json_data = fetch_json_data_for_all_links(links)

        # Already downloaded data for AAPL and GOOGL
        except Exception:
            if company == "AAPL":
                json_data = load_json_data("data/aapl_xbrl_data_final.json")
            if company == "GOOGL":
                json_data = load_json_data("data/googl_xbrl_data_final.json")
            all_revenues = get_all_revenues(json_data)

        st.write("Financial reports downloaded successfully.")

        # User chooses an analysis option
        option = st.selectbox(
            "Choose an option:",
            (
                "Company Overview",
                "Product-Based Revenue Insights",
                "Region-Based Revenue Insights",
            ),
        )
        ## OPTION 1: Company Overview
        if option == "Company Overview":
            sum_csv_filename = f"cache/{company}_summary.csv"
            if os.path.isfile(sum_csv_filename):  # Check if summary CSV file exists
                with open(sum_csv_filename, "r") as file:
                    summary = file.read()
            else:
                summary = sum_bot(company)  # Generate summary using LLM
                with open(sum_csv_filename, "w") as file:
                    file.write(summary)
            # try:
            #     summary = summary.split("Answer:")[1].strip()
            # except IndexError:
            #     pass
            st.write(summary)  # Display the summary

        ## OPTION 2: Product wise Revenue Insights
        elif option == "Product-Based Revenue Insights":
            (
                revenue_product,
                revenue_product_pivot,
                segment_labels,
            ) = get_revenue_product(
                all_revenues, company
            )  # Get revenue insights by product
            visualise_csv_filename = f"cache/{company}_revenue_product_pivot_final.csv"
            revenue_product_pivot.to_csv(visualise_csv_filename)
            if st.checkbox("Visualise"):
                fig = plot_revenue_by_product(
                    revenue_product, revenue_product_pivot, company, segment_labels
                )  # Plot revenue by product
                st.write("Visualization:")
                st.pyplot(fig)

            if st.checkbox("Analyse"):
                analyse_csv_filename = f"cache/{company}_llm_revenue_product.csv"
                if os.path.isfile(
                    analyse_csv_filename
                ):  # Check if analysis CSV file exists
                    st.write("Analysis Result: ")
                    with open(analyse_csv_filename, "r") as file:
                        result = file.read()
                    st.write('<span style="font-family: sans-serif;">' + result + '</span>', unsafe_allow_html=True)  # Display analysis result

                else:
                    result = csv_bot(visualise_csv_filename)  # Analyze data using LLM
                    st.write("Analysis Result:")
                    st.write('<span style="font-family: sans-serif;">' + result + '</span>', unsafe_allow_html=True)
                    with open(analyse_csv_filename, "w") as file:
                        file.write(result)
        ## OPTION 3: Region-Based Revenue Insights
        elif option == "Region-Based Revenue Insights":
            revenue_geo, revenue_geo_pivot, segment_labels = get_revenue_region(
                all_revenues, company
            )  # Get revenue insights by region
            visualise_csv_filename = f"cache/{company}_revenue_geo_pivot_final.csv"
            revenue_geo_pivot.to_csv(visualise_csv_filename)

            if st.checkbox("Visualise"):
                fig = plot_revenue_by_region(
                    revenue_geo, revenue_geo_pivot, company, segment_labels
                )  # Plot revenue by region
                st.write("Visualization:")
                st.pyplot(fig)

            if st.checkbox("Analyse"):
                analyse_csv_filename = f"cache/{company}_llm_revenue_geo_final.csv"
                if os.path.isfile(
                    analyse_csv_filename
                ):  # Check if analysis CSV file exists
                    st.write("Analysis Result: ")
                    with open(analyse_csv_filename, "r") as file:
                        result = file.read()
                    st.write('<span style="font-family: sans-serif;">' + result + '</span>', unsafe_allow_html=True)  # Display analysis result
                else:
                    result = csv_bot(visualise_csv_filename)  # Analyze data using LLM
                    st.write("Analysis Result:")
                    st.write('<span style="font-family: sans-serif;">' + result + '</span>', unsafe_allow_html=True)
                    with open(analyse_csv_filename, "w") as file:
                        file.write(result)

    st.markdown(
        '<h6 style="text-align: center;">Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://dkshjn.github.io/portfolio/">dkshjn</a></h6>',
        unsafe_allow_html=True,
    )  # Made in Streamlit by Daksh Jain


if __name__ == "__main__":
    main()
