import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------


st.set_page_config(
    page_title="Quantitative Portfolio Analyzer",
    page_icon="📈",
    layout="wide"
)


TRADING_DAYS = 252




# --------------------------------------------------
# DATA FUNCTIONS
# --------------------------------------------------


@st.cache_data
def get_prices(tickers, start_date, end_date):
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )


    if len(tickers) == 1:
        prices = data["Close"].to_frame()
        prices.columns = tickers
    else:
        prices = data["Close"]


    return prices.dropna()




def daily_returns(prices):
    return prices.pct_change().dropna()




def annualized_return(returns):
    return returns.mean() * TRADING_DAYS




def annualized_volatility(returns):
    return returns.std() * np.sqrt(TRADING_DAYS)




def max_drawdown(prices):
    running_max = prices.cummax()
    drawdown = prices / running_max - 1
    return drawdown.min()




def sharpe_ratio(annual_return, volatility, risk_free_rate):
    return (annual_return - risk_free_rate) / volatility




def calculate_drawdown(prices):
    running_max = prices.cummax()
    return prices / running_max - 1




def cumulative_returns(returns):
    return (1 + returns).cumprod() - 1




# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------


st.sidebar.title("Portfolio Settings")


ticker_input = st.sidebar.text_input(
    "Assets",
    "SPY, AAPL, JPM, XOM, JNJ, TLT"
)


tickers = [
    ticker.strip().upper()
    for ticker in ticker_input.split(",")
    if ticker.strip()
]


start_date = st.sidebar.date_input(
    "Start Date",
    value=pd.to_datetime("2015-01-01")
)


end_date = st.sidebar.date_input(
    "End Date",
    value=pd.Timestamp.today()
)


risk_free_rate = st.sidebar.slider(
    "Risk-Free Rate",
    0.00,
    0.10,
    0.04,
    0.005
)




# --------------------------------------------------
# HEADER
# --------------------------------------------------


st.title("Quantitative Portfolio & Market Risk Analyzer")


st.markdown(
    """
    Analyze historical returns, volatility, correlations,
    portfolio risk, diversification, and market behavior.
    """
)




# --------------------------------------------------
# DOWNLOAD DATA
# --------------------------------------------------


try:
    prices = get_prices(
        tickers,
        start_date,
        end_date
    )


    returns = daily_returns(prices)


except Exception as e:
    st.error(f"Unable to download market data: {e}")
    st.stop()




# --------------------------------------------------
# METRICS
# --------------------------------------------------


annual_returns = annualized_return(returns)


volatility = annualized_volatility(returns)


drawdowns = max_drawdown(prices)


sharpe = sharpe_ratio(
    annual_returns,
    volatility,
    risk_free_rate
)




# --------------------------------------------------
# TABS
# --------------------------------------------------


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Overview",
        "Asset Analysis",
        "Portfolio Risk",
        "Market Stress"
    ]
)




# ==================================================
# TAB 1 — OVERVIEW
# ==================================================


with tab1:


    st.header("Market Overview")


    metrics = pd.DataFrame({
        "Annual Return": annual_returns,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": drawdowns
    })


    formatted_metrics = metrics.copy()


    formatted_metrics["Annual Return"] = (
        formatted_metrics["Annual Return"]
        .map(lambda x: f"{x:.1%}")
    )


    formatted_metrics["Volatility"] = (
        formatted_metrics["Volatility"]
        .map(lambda x: f"{x:.1%}")
    )


    formatted_metrics["Sharpe Ratio"] = (
        formatted_metrics["Sharpe Ratio"]
        .map(lambda x: f"{x:.2f}")
    )


    formatted_metrics["Max Drawdown"] = (
        formatted_metrics["Max Drawdown"]
        .map(lambda x: f"{x:.1%}")
    )


    st.dataframe(
        formatted_metrics,
        use_container_width=True
    )


    st.subheader("Cumulative Returns")


    cumulative = cumulative_returns(returns)


    st.line_chart(cumulative)




# ==================================================
# TAB 2 — ASSET ANALYSIS
# ==================================================


with tab2:


    st.header("Asset Risk & Return Analysis")


    selected_asset = st.selectbox(
        "Select Asset",
        tickers
    )


    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Annual Return",
        f"{annual_returns[selected_asset]:.1%}"
    )


    col2.metric(
        "Annual Volatility",
        f"{volatility[selected_asset]:.1%}"
    )


    col3.metric(
        "Sharpe Ratio",
        f"{sharpe[selected_asset]:.2f}"
    )


    col4.metric(
        "Max Drawdown",
        f"{drawdowns[selected_asset]:.1%}"
    )


    st.subheader(f"{selected_asset} Price")


    st.line_chart(
        prices[selected_asset]
    )


    st.subheader("60-Day Rolling Volatility")


    rolling_volatility = (
        returns[selected_asset]
        .rolling(60)
        .std()
        * np.sqrt(TRADING_DAYS)
    )


    st.line_chart(
        rolling_volatility
    )


    st.subheader("Historical Drawdown")


    asset_drawdown = calculate_drawdown(
        prices[selected_asset]
    )


    st.line_chart(
        asset_drawdown
    )




# ==================================================
# TAB 3 — PORTFOLIO RISK
# ==================================================


with tab3:


    st.header("Portfolio Risk & Diversification")


    st.markdown(
        "Set the portfolio weights below."
    )


    weights = []


    default_weight = 1 / len(tickers)


    cols = st.columns(
        min(len(tickers), 6)
    )


    for i, ticker in enumerate(tickers):


        with cols[i % len(cols)]:


            weight = st.number_input(
                ticker,
                min_value=0.0,
                max_value=1.0,
                value=float(default_weight),
                step=0.01
            )


            weights.append(weight)


    weights = np.array(weights)


    weight_sum = weights.sum()


    if weight_sum == 0:
        st.warning(
            "Portfolio weights must be greater than zero."
        )


        st.stop()


    weights = weights / weight_sum




    # ----------------------------------------------
    # PORTFOLIO RETURN
    # ----------------------------------------------


    portfolio_return = np.dot(
        weights,
        annual_returns
    )




    # ----------------------------------------------
    # COVARIANCE MATRIX
    # ----------------------------------------------


    covariance_matrix = (
        returns.cov()
        * TRADING_DAYS
    )




    # ----------------------------------------------
    # PORTFOLIO VOLATILITY
    # ----------------------------------------------


    portfolio_variance = (
        weights.T
        @ covariance_matrix.values
        @ weights
    )


    portfolio_volatility = np.sqrt(
        portfolio_variance
    )




    # ----------------------------------------------
    # SHARPE
    # ----------------------------------------------


    portfolio_sharpe = (
        portfolio_return - risk_free_rate
    ) / portfolio_volatility




    # ----------------------------------------------
    # DISPLAY
    # ----------------------------------------------


    col1, col2, col3 = st.columns(3)


    col1.metric(
        "Expected Annual Return",
        f"{portfolio_return:.1%}"
    )


    col2.metric(
        "Annual Volatility",
        f"{portfolio_volatility:.1%}"
    )


    col3.metric(
        "Sharpe Ratio",
        f"{portfolio_sharpe:.2f}"
    )




    st.subheader("Portfolio Allocation")


    allocation = pd.DataFrame({
        "Asset": tickers,
        "Weight": weights
    })


    allocation["Weight"] = (
        allocation["Weight"]
        .map(lambda x: f"{x:.1%}")
    )


    st.dataframe(
        allocation,
        use_container_width=True,
        hide_index=True
    )




    # ----------------------------------------------
    # PORTFOLIO PERFORMANCE
    # ----------------------------------------------


    portfolio_daily_returns = (
        returns
        @ weights
    )


    portfolio_cumulative = (
        (1 + portfolio_daily_returns)
        .cumprod()
        - 1
    )


    st.subheader("Portfolio Growth")


    st.line_chart(
        portfolio_cumulative
    )




    # ----------------------------------------------
    # CORRELATION MATRIX
    # ----------------------------------------------


    st.subheader("Asset Correlations")


    correlation = returns.corr()


    fig, ax = plt.subplots(
        figsize=(8, 6)
    )


    image = ax.imshow(
        correlation,
        vmin=-1,
        vmax=1
    )


    ax.set_xticks(
        range(len(tickers))
    )


    ax.set_yticks(
        range(len(tickers))
    )


    ax.set_xticklabels(
        tickers
    )


    ax.set_yticklabels(
        tickers
    )


    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right"
    )


    for i in range(len(tickers)):
        for j in range(len(tickers)):


            ax.text(
                j,
                i,
                f"{correlation.iloc[i, j]:.2f}",
                ha="center",
                va="center"
            )


    fig.colorbar(image)


    ax.set_title(
        "Return Correlation Matrix"
    )


    st.pyplot(fig)




# ==================================================
# TAB 4 — MARKET STRESS
# ==================================================


with tab4:


    st.header("Market Stress Analysis")


    st.markdown(
        """
        This section examines whether relationships
        between assets change during periods of
        elevated market volatility.
        """
    )


    rolling_vol = (
        returns["SPY"]
        .rolling(60)
        .std()
        * np.sqrt(TRADING_DAYS)
    )


    volatility_threshold = (
        rolling_vol.quantile(0.80)
    )


    stress_days = (
        rolling_vol
        > volatility_threshold
    )


    normal_days = (
        rolling_vol
        <= volatility_threshold
    )


    stress_returns = returns[
        stress_days
    ]


    normal_returns = returns[
        normal_days
    ]


    stress_corr = (
        stress_returns.corr()
    )


    normal_corr = (
        normal_returns.corr()
    )


    st.subheader(
        "Average Equity Correlation"
    )


    equity_tickers = [
        ticker
        for ticker in tickers
        if ticker != "TLT"
    ]


    if len(equity_tickers) > 1:


        stress_equity_corr = (
            stress_corr
            .loc[
                equity_tickers,
                equity_tickers
            ]
        )


        normal_equity_corr = (
            normal_corr
            .loc[
                equity_tickers,
                equity_tickers
            ]
        )


        mask = np.triu(
            np.ones(
                stress_equity_corr.shape,
                dtype=bool
            ),
            k=1
        )


        stress_average = (
            stress_equity_corr
            .where(mask)
            .stack()
            .mean()
        )


        normal_average = (
            normal_equity_corr
            .where(mask)
            .stack()
            .mean()
        )


        col1, col2 = st.columns(2)


        col1.metric(
            "Normal Market Correlation",
            f"{normal_average:.2f}"
        )


        col2.metric(
            "High-Volatility Correlation",
            f"{stress_average:.2f}"
        )




    st.subheader(
        "SPY Rolling Volatility"
    )


    st.line_chart(
        rolling_vol
    )


    st.markdown(
        """
        **Research Question**


        Do diversification benefits decrease when
        markets become more volatile?


        If equity correlations rise during stress
        periods, assets that normally behave
        differently may begin moving together,
        reducing diversification precisely when
        investors need it most.
        """
    )