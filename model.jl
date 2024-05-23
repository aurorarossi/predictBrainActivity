using Graphs, GraphNeuralNetworks, NPZ, LinearAlgebra, Flux, Statistics, Plots, CUDA

#generate dataset

timeseries = npzread("resting-state-100/_100206_timeseries.npy")
timeseries2 = npzread("resting-state-100/_100307_timeseries.npy")

features = []
targets = []
num_timesteps = 3
for i in 1:size(timeseries, 2)-num_timesteps
    push!(features, reshape(timeseries[:, i:i+num_timesteps-1], (1, 100, num_timesteps)))
    push!(targets, reshape(timeseries[:, i+1:i+num_timesteps], (1, 100, num_timesteps)))
end

features2 = []
targets2 = []
for i in 1:size(timeseries2, 2)-num_timesteps
    push!(features2, reshape(timeseries2[:, i:i+num_timesteps-1], (1, 100, num_timesteps)))
    push!(targets2, reshape(timeseries2[:, i+1:i+num_timesteps], (1, 100, num_timesteps)))
end

#complete graph 
g1 = npzread("graph.npy")
#g = SimpleDiGraph(ones(100, 100) - I(100) .+ g1)
g = SimpleDiGraph(g1)
g = GNNGraph(g)

train_loader = [zip(features[1:1000], targets[1:1000]), zip(features2[1:1000], targets2[1:1000])]

test_loader = zip(features[1001:end], targets[1001:end]), zip(features2[1001:end], targets2[1001:end])

model = GNNChain(TGCN(1 => 64), Dense(64, 32, relu), Dense(32, 16, relu), Dense(16, 1))
accuracy(ŷ, y) = 1 - Statistics.norm(y - ŷ) / (Statistics.norm(y) + 0.01)
sqnorm(x) = sum(abs2, x)
loss_function(ŷ, y, m, λ) = Flux.huber_loss(ŷ, y) 

function train(model, train_loader, test_loader,g)
g = gpu(g)
model = gpu(model)
train_loader = gpu(train_loader)
test_loader = gpu(test_loader)
opt = Flux.setup(Adam(0.00001), model)

for epoch in 1:100
    for scan in train_loader
        Flux.reset!(model)
    for (x, y) in scan
        x, y = (x, y)
        grads = Flux.gradient(model) do model
            ŷ = model(g, x)
            loss_function(ŷ, y, model, 0.2)
        end
        Flux.update!(opt, model, grads[1])
    end
end

    if epoch % 10 == 0
        loss = mean([loss_function(model(g, x), y, model, 0.2)  for scan in train_loader for (x, y) in scan])
        train_acc = mean([accuracy(model(g, x), y)  for scan in train_loader for (x, y) in scan])
        test_acc = mean([accuracy(model(g, x), y) for scan in test_loader for (x, y) in scan])
        println("Epoch: $epoch, Loss: $loss, Train Acc: $train_acc, Test Acc: $test_acc")
    end
end
return model
end

model = train(model, train_loader, test_loader,g)
model = cpu(model)

function plot_predicted_data(graph, features, targets, sensor)
    p = plot(xlabel="", ylabel="Brain activity")
    prediction = []
    grand_truth = []
    for i in 1:3:length(features)
        push!(grand_truth, targets[i][1, sensor, :])
        push!(prediction, model(graph, features[i])[1, sensor, :])
    end
    prediction = reduce(vcat, prediction)
    grand_truth = reduce(vcat, grand_truth)
    plot!(p, collect(1:length(features)), grand_truth, color=:blue, label="Grand Truth", xticks=([i for i in 0:50:250], ["$(i)" for i in 0:4:24]))
    plot!(p, collect(1:length(features)), prediction, color=:red, label="Prediction")
    return p
end

plot_predicted_data(g, features[1000:1179], targets[1000:1179], 4)
display(plot_predicted_data(g, features[1000:1179], targets[1000:1179], 4))
display(plot_predicted_data(g, features2[1000:1179], targets2[1000:1179], 4))