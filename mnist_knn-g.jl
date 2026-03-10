################################################################################
#=
Install the required packages with:
using Pkg
Pkg.add("MLDatasets")
Pkg.add("Random")
Pkg.add("Dates")
Pkg.add("NearestNeighbors")
Pkg.add("Plots")
=#
using MLDatasets
using Random
using Dates
using NearestNeighbors
using Plots

#= 
The MNIST dataset is not just one pile of 70,000 images. 
The original creators (Yann LeCun et al.) officially divided it into two separate buckets
to ensure everyone benchmarks their models the same way:
60000 assigned to the training set and 10000 
=#

#=
 K-NN algorithm based on  a neighbor search structure constructed with a  BruteTree, that is
computing all distances on the fly without any spatial partitioning.
And computing the neighbors for all test points in one batch, up to the maximum K we want to evaluate,
so we can compute accuracies for multiple K values in one pass.
=#

println("Loading MNIST...")
train_x_raw, train_y_raw = MNIST.traindata()
test_x_raw,  test_y_raw  = MNIST.testdata()

# train_data = MNIST(split=:train)
# train_x_raw, train_y_raw = MNIST(split=:train)[:]
# test_x_raw,  test_y_raw  = MNIST(split=:test)[:]

LIMIT_TRAIN = 20_000
LIMIT_TEST  = 2_000

#784 pixels as 1d array features
function flatten_images_subset(images)::Matrix{Float32}
    n_samples = size(images, 3)
    # reshape -> (784, n), then transpose -> (n, 784) so rows are images
    X = reshape(images, 28*28, n_samples)
    return Float32.(transpose(X)) ./ 255f0  # normalize 0..255 to 0..1
end

#creates a view of the data instead of copying it
X_train = flatten_images_subset(@view train_x_raw[:, :, 1:LIMIT_TRAIN])
X_test  = flatten_images_subset(@view test_x_raw[:,  :, 1:LIMIT_TEST])

y_raw_subset = copy(train_y_raw[1:LIMIT_TRAIN])
y_test       = test_y_raw[1:LIMIT_TEST]

# noise is added to the training labels
println("Corrupting  training labels to simulate noise...")
rng = Random.MersenneTwister(3)
n_corrupt = Int(floor(0.13 * length(y_raw_subset)))
# pick images to be corrupted
idxs = shuffle(rng, 1:length(y_raw_subset))[1:n_corrupt]

y_raw_subset[idxs] = rand(rng, 0:9, n_corrupt)

# --- VISUALIZATION BLOCK START ---
println("Displaying mixed examples (Corrupted & Clean)...")

corrupted_indices = idxs[1:5]

all_indices = 1:length(y_raw_subset)
clean_indices = setdiff(all_indices, idxs)[1:4]

sample_indices = [corrupted_indices; clean_indices]
my_plots = []

for i in sample_indices
    # Get the image
    img_matrix = train_x_raw[:, :, i]'

    # Get labels
    real_label = train_y_raw[i]
    fake_label = y_raw_subset[i]

    if i in idxs
        title_text = "LIE: Is $real_label -> Says $fake_label"
    else
        title_text = "OK: Is $real_label -> Says $fake_label"
    end

    p = heatmap(img_matrix, yflip=true, c=:grays, legend=false, axis=false,
                title=title_text, titlefontsize=9)
    push!(my_plots, p)
end

display(plot(my_plots..., layout=(3,3), size=(700,700)))
println("Plot displayed! First 5 are Lies, last 4 are Clean.")
sleep(5)
# --- VISUALIZATION BLOCK END ---

#=
the distance d(x,y) is compute as euclidiean distance between the feature vectors of the images:
d(x,y) = sqrt(sum((x_i - y_i)^2 for i in 1:length(x)))
where x_i and y_i are the pixel values of the two images being compared.
Each x_i is a value between 0 and 255, representing the intensity of a pixel in the image,
normalized to a range of 0 to 1 in our case.
The KNN algorithm uses this distance to find the k nearest neighbors of a given test image and makes predictions based on the majority class among those neighbors. 
=#

println("\n")
println("Testing the effect of k with Noisy Labels")
println("\n")

# Train: for knn, this just organize the data for fast neighbor lookup. A structured map of the data.
#=
build the neighbor search structure ONCE, query neighbors ONCE up to maxK,
and compute accuracies for multiple K values in one pass.
(With BruteTree, "training" is basically storing the points; the big win is avoiding repeated work.)
=#

k_list = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51, 101, 201, 401, 801]
maxK = maximum(k_list)

# NearestNeighbors expects : (dim, npoints)
train_pts = permutedims(X_train)  # (784, n_train)
test_pts  = permutedims(X_test)   # (784, n_test)

start_time = now()

tree = BruteTree(train_pts)

# _ ignores distances
idxs_raw, _ = knn(tree, test_pts, maxK, true)

nn_idxs = if eltype(idxs_raw) <: Integer
    Int.(idxs_raw)  
else
    println("Warning: knn returned a vector of vectors. Reshaping to (maxK, n_test).\n")
    # transform into a matrix
    Int.(reduce(hcat, vec(idxs_raw)))  # => (maxK, n_test)
end

n_test = size(nn_idxs, 2)    

correct = zeros(Int, length(k_list))
counts = zeros(Int, 10)  

for j in 1:n_test
    fill!(counts, 0)
    best_label = 1
    best_score = 0
    kpos = 1

    # neighbors are computed only once up to maxK, so we can evaluate multiple K values in one pass 
    for t in 1:maxK
        idx = nn_idxs[t, j]
        label = y_raw_subset[idx] + 1       # shift to 1..10
        c = (counts[label] += 1)
        # the final label, in case of tie, will be the one that reached the count first
        if c > best_score
            best_label = label
            best_score = c
        end

        if kpos <= length(k_list) && t == k_list[kpos]
            if (best_label - 1) == y_test[j]
                correct[kpos] += 1
            end
            kpos += 1
        end
    end
end

for (i, k) in enumerate(k_list)
    acc = correct[i] / n_test
    println("k = $k \t Accuracy: $(round(acc * 100, digits=2))%")
end

end_time = now()
duration_seconds = Dates.value(end_time - start_time) / 1000.0

println("\n")
println("Total Duration:   $(round(duration_seconds, digits=4)) seconds")
println("\n")
