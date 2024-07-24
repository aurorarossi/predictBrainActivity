function Z_score_normalization(x; dims=2)
    return (x.- mean(x, dims=dims))./ std(x, dims=dims)
end

function create_features_targets(x; num_timesteps=3)
    features = []
    targets = []
    num_timesteps = 3
    for i in 1:size(x, 2)-num_timesteps
        push!(features, reshape(x[:, i:i+num_timesteps-1], (1, 100, num_timesteps)))
        push!(targets, reshape(x[:, i+1:i+num_timesteps], (1, 100, num_timesteps)))
    end
    return features, targets
end

function load_graph(filename)
    g = npzread(filename)
    g = SimpleDiGraph(g)
    g = GNNGraph(g)
    return g
end   

function plot_predicted_data(graph, features, targets, sensor)
    p = plot(xlabel="Time step", ylabel="Brain activity (BOLD)", title="Sensor $(sensor)")
    prediction = []
    grand_truth = []
    for i in 1:3:length(features)
        push!(grand_truth, targets[i][1, sensor, :])
        push!(prediction, model(graph, features[i])[1, sensor, :])
    end
    prediction = reduce(vcat, prediction)
    grand_truth = reduce(vcat, grand_truth)
    plot!(p, collect(1:length(features)), grand_truth, color=:blue, label="Ground Truth", xticks=([i for i in 0:50:250], ["$(i)" for i in 0:4:24]), lw=2)
    plot!(p, collect(1:length(features)), prediction, color=:red, label="Prediction", lw=2)
    return p
end