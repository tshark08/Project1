from mixtral import MixtralModel

# Load the Mixtral model
model = MixtralModel.from_pretrained("mixtral/mixtral-base")

# Define the prompt
prompt = """
<s>
Given a text description of an issue, classify it into one of the following candidate labels: Payroll, Tax, System, Benefits, Support, Banking, Training, Miscellaneous.

Example 1:
Input: ["Mutiple Tax Fililing and Withholding issues"], Keywords -['filing', 'irs','tax','pay','wages','withholding']
Output: Tax

Example 2:
Input: ["Payroll and Time Accrual Issues"], Keywords -- ['payroll',overtime','accurals, 'timesheet','pto','employees']
Output: Payroll

Example 3:
Input: ["System Upgrade and Integration Issues"], Keywords -- ['upgrade','integration','system','software','deployment']
Output: System

Input: {}
</s>
"""

# Load data
data = pd.read_csv('data.csv')

# Combine Topic Representation and Keywords columns
data['Text'] = data['Topic Representation'] + ' ' + data['Keywords'].apply(lambda x: ' '.join(x))

# Make predictions
predictions = []
for text in data['Text']:
    input_prompt = prompt.format(text)
    output = model(input_prompt)
    label = output.split("Output: ")[-1].strip()
    predictions.append(label)

# Save predictions
data['Predicted Label'] = predictions
data.to_csv('data_with_predictions.csv', index=False)
