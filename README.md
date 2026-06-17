# DataSense — CSV & Dataset Analyzer

**DataSense** is an intelligent Streamlit web application that transforms CSV file analysis. Upload any CSV, and instantly explore it with AI-powered insights, interactive visualizations, and a natural language chat interface.

![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python)
![Anthropic](https://img.shields.io/badge/Anthropic-Claude-000000?style=for-the-badge)

---

## 🎯 Features

### Two Analysis Modes

- **⚡ Quick Insights** — Plain English summaries, clean charts, simple Q&A for non-technical users
- **🧠 ML Analysis** — Data quality scoring, correlation heatmaps, model suggestions, and technical pandas-powered chat

### Interactive Visualizations

- Distribution histograms with KDE overlays
- Correlation heatmaps with color-coded cells
- Missing value analysis by column
- Categorical value counts (top 10)
- Scatter matrices for numeric features
- Real-time chart generation from natural language queries

### AI-Powered Analysis

- Natural language summaries using Claude 3.5 Sonnet
- Chat with your data — ask questions in plain English or pandas code
- Automatic target variable detection
- ML model recommendations based on your dataset
- Automatic pandas code generation for complex queries

### Data Quality Assessment

- Quality score (0-100) with detailed breakdowns
- Duplicate detection
- Missing value analysis by percentage
- Cardinality analysis for categorical features
- Memory usage estimation

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- An Anthropic API key (get one at [console.anthropic.com](https://console.anthropic.com))

### Installation

1. **Clone or download the project**

   ```bash
   cd DataSense
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API key**

   Create a `.env` file in the project root:

   ```env
   ANTHROPIC_API_KEY=sk-ant-...your-key-here...
   ```

   Or set it as an environment variable:

   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...your-key-here...
   ```

5. **Run the app**

   ```bash
   streamlit run app.py
   ```

   The app opens at `http://localhost:8501`

---

## 📖 How to Use

### Stage 1: Upload Your CSV

1. Open the app and upload any CSV file
2. DataSense scans the file and shows key statistics:
   - Row and column counts
   - Column types (numeric, categorical, datetime, boolean)
   - Data quality score
   - Memory usage

### Stage 2: Choose Your Mode

- **Quick Insights** — For exploration and simple Q&A
- **ML Analysis** — For deeper technical analysis and model planning

The app generates a tailored summary based on your chosen mode.

### Stage 3: Explore and Chat

**Dashboard Tabs:**

- **📈 Summary** — AI-generated overview of your dataset
- **📊 Charts** — Interactive visualizations (distributions, correlations, scatter plots)
- **💬 Chat with Data** — Ask questions and get instant answers
- **🧠 ML Insights** (ML mode only) — Model recommendations and feature analysis

**Chat Examples:**

_Quick Mode:_
- "What does this data represent?"
- "Are there any obvious patterns?"
- "Which columns have the most missing data?"

_ML Mode:_
- "What's the correlation between age and income?"
- "Group by category and show mean values"
- "Show me the distribution of the target variable"

---

## 📁 Project Structure

```
DataSense/
├── app.py                      # Main Streamlit app
├── requirements.txt            # Dependencies
├── .env                        # API keys (not committed)
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
└── utils/
    ├── __init__.py
    ├── scanner.py              # CSV scanning & analysis
    ├── visualizer.py           # Plotly chart generation
    ├── llm.py                  # Claude API integration
    └── chat.py                 # Code execution & chart conversion
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key |

### Streamlit Cloud Deployment

To deploy on Streamlit Cloud:

1. Push your code to a GitHub repo (without `.env`)
2. Create a new app on [share.streamlit.io](https://share.streamlit.io)
3. Select your repo and branch
4. Go to **Settings → Secrets** and add:
   ```
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```

The app automatically detects and uses Streamlit secrets.

---

## 📊 Data Quality Scoring

The quality score (0–100) is calculated as:

- Start at 100
- Subtract 5 per column with >5% missing values
- Subtract 10 per column with >30% missing values
- Subtract 5 if duplicates > 1% of rows
- Subtract 3 if any column has only one unique value (constant)
- Clamp between 0 and 100

---

## 💡 Usage Tips

### Best Practices

1. **Clean your data first** — Remove or handle obvious errors before uploading
2. **Name columns clearly** — The AI works better with descriptive column names
3. **Use consistent datatypes** — Avoid mixing numbers and text in the same column
4. **For ML analysis** — Use datasets with clear target variables when possible

### Limitations

- Maximum file size: 50 MB (larger files are warned and sampled to 10,000 rows)
- Maximum rows for full analysis: 10,000 (for performance)
- Requires at least 2 numeric columns for correlation heatmaps
- Chat code execution is sandboxed — only pandas, numpy, and basic operations are available

---

## 🎨 Customization

### Styling

The app uses a dark theme with custom colors:

- **Primary Accent:** `#6366f1` (Indigo)
- **Success:** `#10b981` (Emerald)
- **Background:** `#0f0f13` (Near-black)
- **Cards:** `#1a1a24` (Dark gray)

Edit the CSS in `app.py` to customize colors and styling.

### Models

To use a different Claude model, edit `llm.py`:

```python
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",  # Change this
    ...
)
```

Available models: `claude-3-opus-20250219`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "API key not found" | Ensure `.env` file exists or `ANTHROPIC_API_KEY` is set |
| File upload fails | Verify CSV format; try opening in Excel first |
| Chat not working | Check API key validity and Anthropic account quota |
| Charts not showing | Ensure you have enough numeric columns for the chart type |
| App is slow | The file may be too large; try sampling the data |

---

## 🚀 Roadmap (Phase 2)

Future enhancements planned:

- 📄 Export reports as PDF
- 🔀 Column comparison tool
- ⚙️ Automated feature engineering with code suggestions
- 🔗 Dataset merge/join utilities
- 📈 Time series detection and forecasting
- 💾 Save and reload analysis sessions

---

## 📝 License

This project is open source. Feel free to modify and distribute.

---

## 🤝 Contributing

Found a bug or have a feature request? Open an issue or submit a pull request!

---

## 📞 Support

For questions about:

- **Streamlit** — [Streamlit Docs](https://docs.streamlit.io)
- **Anthropic Claude API** — [Anthropic Docs](https://docs.anthropic.com)
- **Plotly** — [Plotly Docs](https://plotly.com/python)
- **Pandas** — [Pandas Docs](https://pandas.pydata.org/docs)

---

**Built with ❤️ using Streamlit, Anthropic Claude, and Plotly**
