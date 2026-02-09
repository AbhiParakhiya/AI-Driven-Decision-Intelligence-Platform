import os

class LLMInsights:
    def __init__(self, api_key=None, provider="mock"):
        self.api_key = api_key
        self.provider = provider

    def generate_insight(self, context_data):
        """
        Generates business insights based on the provided context data.
        """
        sales_forecast = context_data.get('forecast', 0)
        current_inventory = context_data.get('inventory', 0)
        trend = context_data.get('trend', 'stable')
        region = context_data.get('region', 'All')
        category = context_data.get('category', 'All')
        
        # Construct a prompt for the LLM
        prompt = f"""
        act as a senior data analyst.
        Analyze the following S&OP data:
        - Region: {region}
        - Category: {category}
        - Forecasted Sales for next period: {sales_forecast} units
        - Current Inventory: {current_inventory} units
        - Trend: {trend}
        
        Provide:
        1. A brief summary of the situation.
        2. A risk assessment (Stockout or Overstock).
        3. A recommended action.
        """
        
        if self.provider == "mock":
            return self._mock_response(sales_forecast, current_inventory, trend)
        else:
            # Placeholder for actual API call (OpenAI/HuggingFace)
            return f"LLM API response for: {prompt}"

    def _mock_response(self, forecast, inventory, trend):
        """
        Rule-based logic to simulate LLM insights.
        """
        insight = "### Business Insight Report\n\n"
        
        # Summary
        insight += f"**Observation:** Demand is expected to be **{forecast:.0f} units**, with current inventory at **{inventory} units**.\n\n"
        
        # Risk Assessment
        gap = inventory - forecast
        if gap < 0:
            insight += "Risk Alert: Potential **Stockout** detected. Supply is insufficient to meet projected demand.\n\n"
            action = "Increase procurement orders immediately to cover the deficit."
        elif gap > forecast * 0.5:
            insight += "Risk Alert: Potential **Overstock** detected. Inventory levels are significantly higher than demand.\n\n"
            action = "Consider running a promotion to clear excess stock."
        else:
            insight += "Status: Inventory levels are **Optimal** and aligned with forecasted demand.\n\n"
            action = "Maintain current supply chain settings."
            
        # Recommendation
        insight += f"**Recommendation:** {action}"
        
        return insight

if __name__ == "__main__":
    # Test
    llm = LLMInsights(provider="mock")
    ctx = {'forecast': 150, 'inventory': 100, 'trend': 'up', 'region': 'North', 'category': 'Electronics'}
    print(llm.generate_insight(ctx))
