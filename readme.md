# Fake News Detector

This system provides a comprehensive analysis of news headlines to determine their likely credibility. It is designed to help users identify potential misinformation by evaluating a headline from multiple perspectives.

## Core Functionality

The system takes a news headline as input and processes it through a series of specialized analysis modules. Each module evaluates a different aspect of the headline and its source. The results are then aggregated to produce a final credibility score and a clear verdict.

## Analysis Components

The final verdict is based on a combination of the following analyses:

### 1. Clickbait Detection

This module examines the language and structure of the headline itself. It identifies patterns commonly used in clickbait—such as sensationalism, emotional manipulation, or intentionally withholding information—to score how "clickbait-y" the headline is. A high clickbait score can indicate that the content is more focused on generating clicks than presenting factual information.

### 2. Source Credibility Analysis

This component evaluates the trustworthiness of the news source's domain (e.g., `nytimes.com`, `yournews.co`). It checks the domain against databases of known reliable sources, biased publishers, and fake news sites to assess the general reputation of the publisher.

### 3. Claim Verification

This module fact-checks the central claim of the headline. It searches the web for other articles, reports, and fact-checking websites that have covered the same topic. By comparing the headline to multiple independent sources, it determines whether the claim is supported by evidence or contradicted by other reports.

### 4. Network Propagation Analysis

This analysis looks at how the story is spreading across the internet. It investigates the diversity and quality of the domains reporting on the story. A credible story is typically picked up by many different and reputable news outlets, while misinformation is often confined to a small network of unreliable sites.

## The Final Verdict

By combining the scores from these four modules, the system calculates a final credibility rating. This is presented to the user as a simple verdict, such as `Credible`, `Doubtful`, or `False`, along with a detailed breakdown of each analysis component, giving you a transparent look at why the headline received its rating.
