# import Pkg
# Pkg.add(["MLJ", "NearestNeighborModels", "DataFrames"])

using MLJ
using DataFrames

# 1. Load the Data
# MLJ has built-in access to standard datasets
X, y = @load_iris

# 2. Split the Data
# We partition the data: 70% for training, 30% for testing
train_rows, test_rows = partition(eachindex(y), 0.7, shuffle=true, rng=42)

# 3. Load the Model
# We load the KNNClassifier from the 'NearestNeighborModels' package
KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels
model = KNNClassifier(K=3) # We choose to look at the 3 nearest neighbors

# 4. Create and Train the Machine
# In MLJ, a 'machine' binds the model to the data
mach = machine(model, X, y)
fit!(mach, rows=train_rows)

# 5. Make Predictions
# predict_mode gives the class label (e.g., "setosa") rather than probabilities
y_hat = predict_mode(mach, rows=test_rows)

# 6. Evaluate Accuracy
# Compare predictions (y_hat) against the actual values (y[test_rows])
acc = accuracy(y_hat, y[test_rows])

println("---------------------------------------------------")
println("Model Accuracy on Test Set: $(round(acc * 100, digits=2))%")
println("---------------------------------------------------")

# Optional: View a confusion matrix to see where errors happened
cm = confusion_matrix(y_hat, y[test_rows])
display(cm)

# In your Julia script
stats = @timed fit!(mach, rows=train_rows)
println("Training Time: $(stats.time) seconds")
