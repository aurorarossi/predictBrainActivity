using Graphs, GraphNeuralNetworks, NPZ, LinearAlgebra, Flux, Statistics, Plots, CUDA
include("utils.jl")

#generate dataset

timeseries1 = npzread("data/resting-state-100/_100206_timeseries.npy")
timeseries2 = npzread("data/resting-state-100/_100307_timeseries.npy")

# Z-score normalization of the timeseries
timeseries1 = Z_score_normalization(timeseries1)
timeseries2 = Z_score_normalization(timeseries2)

features1, targets1 = create_features_targets(timeseries1)
features2, targets2 = create_features_targets(timeseries2)

#create graph
g = load_graph("data/graph.npy")

train_loader = [zip(features1[1:1000], targets1[1:1000]), zip(features2[1:1000], targets2[1:1000])]
test_loader = [zip1(features[1001:end], targets1[1001:end]), zip(features2[1001:end], targets2[1001:end])]

model = GNNChain(
                TGCN(1 => 64), 
                Dense(64, 32, relu), 
                Dropout(0.2), 
                Dense(32, 16, relu), 
                Dropout(0.2), 
                Dense(16, 1))

accuracy(ŷ, y) = 1 - (Statistics.norm(y - ŷ) / (Statistics.norm(y) + 0.01))
loss_function(ŷ, y) = Flux.huber_loss(ŷ, y) 

function train(model, train_loader, test_loader,g)
    g = gpu(g)
    model = gpu(model)
    train_loader = gpu(train_loader)
    test_loader = gpu(test_loader)

    opt = Flux.setup(OptimiserChain(WeightDecay(),Adam(0.00001)), model)

    for epoch in 1:50
        for scan in train_loader
            Flux.reset!(model)
            for (x, y) in scan
                x, y = (x, y)
                grads = Flux.gradient(model) do model
                    ŷ = model(g, x)
                    loss_function(ŷ, y)
                end
                Flux.update!(opt, model, grads[1])
            end
        end

        if epoch % 10 == 0
            loss = mean([loss_function(model(g, x), y)  for scan in train_loader for (x, y) in scan])
            train_acc = mean([accuracy(model(g, x), y)  for scan in train_loader for (x, y) in scan])
            test_acc = mean([accuracy(model(g, x), y) for scan in test_loader for (x, y) in scan])
            println("Epoch: $epoch, Loss: $loss, Train Acc: $train_acc, Test Acc: $test_acc")
        end
    end
    return model
end

model = train(model, train_loader, test_loader,g)
model = cpu(model)


display(plot_predicted_data(g, features1[1000:end], targets1[1000:end], 60))
display(plot_predicted_data(g, features2[1000:end], targets2[1000:end], 60))