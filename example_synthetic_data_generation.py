import pandas as pd
from AutoLLM.modules.synthetic_data_agent import SyntheticDataAgent

# Load the ecommerce dataset
df = pd.read_csv('data/Ecommerce/ecommerceDataset.csv')

# Extract text samples
text_samples = df['Text'].dropna().tolist()

# Initialize the synthetic data agent
agent = SyntheticDataAgent()

# Generate synthetic data
synthetic_data = agent.generate_synthetic_data(text_samples[:100], num_samples=10)

# Save results
output_df = pd.DataFrame({
    'Original_Text': text_samples[:10],
    'Synthetic_Text': synthetic_data
})
output_df.to_csv('synthetic_ecommerce_data.csv', index=False)

print("Generated 10 synthetic examples:")
print(output_df)