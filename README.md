# resume_projects
A collection of projects on my resume

Project 1: Horse Racing Outcome Prediction Model (Early Version):
This version contains the first successful convergence I achieved using maximum likelihood estimation (MLE) for predicting horse race outcomes. It predates the integration of the explosion technique from academic literature, which later improved model accuracy to ~42%. Current iterations (not included here) are incorporating jockey data in an effort to surpass the 50% accuracy threshold. This was the first project that got me interested in ML concepts. The earliest versions I used matlab and "converted the logistic regression model to linear algebra" in an effort to do it by hand. That method provided a great foundation for understanding the limitations and errors teat may arise witht he model.

Project 1 2026 Update:  
  I modernized the data pipeline using AI agents to solve the bottleneck of manual data entry. Previously, it took 1.5 months to translate 200 race samples by hand into Excel, but by developing three agents—one for horse data, one for jockey data, and one for race translation—I reduced the process to a few hours for 250 samples. This served as a proof-of-concept for using agentic workflows in data aggregation.
  I also updated the model architecture, moving from multinomial logistic regression to an XGBRanker model. While logistic regression works for binary structures, the ranker is better suited for training on individual race shapes where you must compare the winner to other horses within that specific race rather than a global pool of losers.
  A key breakthrough in my research was correcting how the model interacts with public odds. Initially, I mistakenly trained the model with public odds included, but I have since shifted to a Benter-style framework where the model is a linear combination: $Model\_{odds} + Public\_{odds}$. After properly building this per the research, I found the model provides little edge because the public is already ~50% accurate at picking the winner.
  
This led to two major realizations:
  1. The model is highly effective at picking the 2nd and 3rd horses; accuracy increases significantly if you assume the public's first pick might be wrong.
  2. Public odds already bake in the numerical data and insider knowledge my model relies on.

To create a true edge, I am now moving toward non-traditional data. By extracting natural language track notes (e.g., "opened up," "ridden out," "tired") from race docs, I can generate signals that aren't purely numerical. This shift inspired my work on a multi-layer political prediction system using semantic NLP pipelines and RAG-based text parsing to forecast outcomes before they reach market consensus.
  

Project 2: Qunatitative beta-constrained long-short portfolio optimization:
Built a beta-constrained long-short equity portfolio over a 2-day sprint for a firm’s evaluation project. Designed a market-neutral strategy using historical data from Jan 2022 to mid-2025, optimizing for risk-adjusted returns while maintaining low beta exposure. Delivered code, a one-page report, and a recorded walkthrough of the methodology and results.

Project 3: Quantitative Equities Screener Tool - TBA 
