---
layout: article
title: Valuation
comments: true
categories: finance
image:
  teaser: jupyter-main-logo.svg
---

Buffett is known for his unrivaled ability to pick "wonderful companies at a fair price."  Wonderful has a qualitative and a quantitative aspect. 


## Greenbalt's method of quantifying 'wonderful'

Greenblatt translated 'wonderful' to Return on Capital (or ROC) in his book 'The Little Book That Beats The Market'. The higher the ROC, the more wonderful the company:

'ROC =  EBIT / tangible capital (the cash + the assets of the company)'

As for fair price, Greenblatt translated fair price to earnings yield (acquirer's multiple). the higher the earning yield, the better the price:


'earnings yield = EBIT / enterprise value'


Tobias Carlisle and Wes Gray, Ph.D. these metrics in their book, Quantitative Value. Testing it back to 1964 (until 2004), they found that, a portfolio of companies with highest value of both these metrics among the top 3400 companies, outpaced the market at a 13.9% return rate (compared to 10.5% for the S&P 500). When tested out separately, the earnings yield of the selected stocks returned 16%, while the ROC actually underperformed compared to the S&P. After more research, they concluded that simply using the earnings yield proved to be the most efficient way to find undervalued stocks. 


Quantitative Value is a model-based investment strategy that uses above metrics to generate a portfolio that is strictly followed and re-balanced frequently (quarterly or annually). 


## Buffet's way of deciding the price worth paying for a stock

- The value of a business today is equal to the net profits that it generates in future plus the value of its assets (book value). However, if we pay that price today, we are not going to get any returns. Warren Buffet likes to discount that price by the risk-free interest rate of 10-year bonds, to get to the intrinsic value of a business. 
	+ You'd want to buy at a very good discount compared to the intrinsic value price or see a very promising growth trajectory that is better than a linear growth assumed in the intrinsic value to justify investing in a company. 
		* Alternatively, we can discount that future value by a minimum growth rate we'd like, back to today, to arrive at a price that makes sense for us to buy in. 


- The most important rule of investing is protecting the principle and not losing money. That's why having a huge margin of safety in case of worst case scenario is important. This is possible by understanding that risk and uncertainty are two different things that are usually both priced into the business. Interesting businesses are the ones that have minimal risk but have a high degree of uncertainty, therefore, are available at very attractive prices.
	+ Risk is how much of your principle you will lose in the worst case scenario. You should not lose principle!
	+ Uncertainty is not knowing how much upside of growth you get. 

- Warren Buffet actually has a few pillars that he looks into when deciding whether to invest in a business. He likes to think of a company as a castle with management as the knight in charge of it. 
	1. the first one is vigillant leadership. This has qualitative AND quantitative aspects.
		* quantitatively, it's about the ratio of money owed to assets and money coming in compared to money going out.
			+ Debt/Equity ratio: the ratio of company debt to company assets. He likes companies with Debt/Equity ratio below .50. Meaning, if a company makes $10, it only has to pay $5 in debt.
			+ The Current Ratio: the ratio of money coming in in the next 12 months comparted to the money going out. He likes companies that are above 1.50. Meaning every $10 going out on cash, $15 should be cash inflow. 
		* qualitatively, it's about whether the management of the company is careful and somewhat paranoid about potential threat to the company. 
	2. Understandability and stability: He only invests in companies where he understands the business with somewhat consistent and stable record to make prediction of future performance of the business easier. He'd like to be able to understand the weaknesses and strengths of the castle and be able to predict the future performance of the castle.
	3. Long term prospects: He'd like to evaluate if the business model is something that will work in future. He likes castles that will endure future. 
	4. Competative advantage (moat): He likes to see companies that have a sort of competetive advantage that keeps others away like the moat around the castle. 
	5. Undervalued stock: Warren's first rule of investing is protect your principle. He likes to invest in companies that he can buy at a price that gives him a huge margin of safety in case of error in evaluating the company or other unpredictable future adversities. That way, worst case scenario, he should be able to get his principle money back by liqidation of the assets of the company.





- Warren buffet likes to compare investing in companies to government issued bonds, since bonds returns are risk-free given that government can always print new money. Therefore, he discounts the future value of a company 10 years in future by the interest rate of a 10-year bond. That would be the price which if he buys into the company at, he'd get return equal to 10-year bonds. Therefore, to get a return that justifies the risk of investing in a company compared to bonds, he would need to get a much better price. 

- Mohnish Pabrai that learned from Warren to a successful investment career says that he'd only look at companies where the market price is at least two times less than the intrinsic value to give himself a huge margin of safety. 



We need to estimate several things to calculate the intrinsic value of a business and the price at which it makes sense for us to buy in. 
	+ Future value: use the average growth rate in the past to linearly estimate it for example in 10 years in future. Be careful about high growth rate in an extended amount of time in future. very few companies grow at 10% for a sustained amount of time (e.g. 10 yrs). That's why you need to assess the market conditions, maturity of the industry, competitors and etc while determining the growth rate.  
		* not all of the future free cash flow will be reflected in the value of the business since there may be efficiency issues in the company. 
		* Therefore, we can instead use the average book value growth rate and estimate future book value plus the sum of dividends that the company pay during this time. This is the future value of the business 
		* we need to check how good/conservative this estimate is by seeing if 
			- the average growth rate makes sense
			- if linear estimation is conservative enough
	+ Intrinsic value: discount the future value to today by the 10-year bond interest rate. If we use the 10-year bonds interest rate to discount the future value company to today, the price that we get for today will be the price that if we buy the company at, our money will grow similar to a 10-year bonds. Therefore, since government bonds are almost risk-free, and the business may have risk, at that price, buying into the business doesn't make sense.
 


 - The formula that is used to calculate the value of a bond in future or the intrensic value of a stock is:

 $$ price = C * \frac{[1- [\frac{1}{(1+i)^n}]]}{i} + \frac{M}{(1+i)^n} $$

where: 

M = Future value of company (or bond value at maturity)
C = yearly dividend payment over n years (or bond coupon payment)
i = the interest rate in %
n = the number of years in future


Model & Random && Artist & Politician & Writer& Scientist && Plant & CelestialBody\\ \hline
% Clique-Discr. (7) & 73.86 && 00.00 & 00.00 & 00.00 & 71.86 && 65.47 & 61.44 \\
Clique-Discr. (3) & 76.17 && 73.01 & 73.82 & 73.28 & 74.56 && 66.14 & 60.38 \\