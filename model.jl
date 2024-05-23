using Graphs, GraphNeuralNetworks, NPZ, LinearAlgebra, Flux, Statistics, Plots

#generate dataset

timeseries = npzread("resting-state-100/_100206_timeseries.npy")

features = []
targets = []
num_timesteps = 3
for i in 1:size(timeseries,2)-num_timesteps
    push!(features, reshape(timeseries[:,i:i+num_timesteps-1], (1,100,num_timesteps)))
    push!(targets, reshape(timeseries[:,i+1:i+num_timesteps], (1,100,num_timesteps)))
end

#complete graph 
g1 = npzread("graph.npy")
g = SimpleDiGraph(ones(100,100)-I(100).+g1)
g = GNNGraph(g)

train_loader = zip(features[1:400], targets[1:400])

test_loader = zip(features[1001:end], targets[1001:end])

model = GNNChain(TGCN(1 => 64), Dense(64, 32,relu), Dense(32,16,relu), Dense(16,1))
accuracy(ŷ, y) = 1 - Statistics.norm(y-ŷ)/Statistics.norm(y)
loss_function(ŷ, y) = Flux.mae(ŷ, y)

sqnorm(x) = sum(abs2, x)

julia> sum(sqnorm, Flux.params(m))

function train(g, train_loader,test_loader, model)
  

    opt = Flux.setup(Adam(0.001), model)

    for epoch in 1:100
        for (x, y) in train_loader
            x, y = (x, y)
            grads = Flux.gradient(model) do model
                ŷ = model(g, x)
                Flux.mae(ŷ, y) 
            end
            Flux.update!(opt, model, grads[1])
        end
        
        if epoch % 10 == 0
            loss = mean([Flux.mae(model(g,x), y) for (x, y) in train_loader])
            train_acc = mean([accuracy(model(g,x), y) for (x, y) in train_loader])
            test_acc = mean([accuracy(model(g,x), y) for (x, y) in test_loader])
            println("Epoch: $epoch, Loss: $loss, Train Acc: $train_acc, Test Acc: $test_acc")
        end
    end
    return model
end

model = train(g, train_loader,test_loader, model)

function plot_predicted_data(graph,features,targets, sensor)
    p = plot(xlabel="Time (h)", ylabel="Brain activity")
    prediction = []
    grand_truth = []
    for i in 1:3:length(features)
        push!(grand_truth,targets[i][1,sensor,:])
        push!(prediction, model(graph, features[i])[1,sensor,:]) 
    end
    prediction = reduce(vcat,prediction)
    grand_truth = reduce(vcat, grand_truth)
    plot!(p, collect(1:length(features)), grand_truth, color = :blue, label = "Grand Truth", xticks =([i for i in 0:50:250], ["$(i)" for i in 0:4:24]))
    plot!(p, collect(1:length(features)), prediction, color = :red, label= "Prediction")
    return p
end

plot_predicted_data(g,features[1000:1179],targets[1000:1179], 2)