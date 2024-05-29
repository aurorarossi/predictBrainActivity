using GraphNeuralNetworks, Flux
struct GConvLSTMCell <: GNNLayer
    conv_x_i::ChebConv
    conv_h_i::ChebConv
    w_i
    b_i
    conv_x_f::ChebConv
    conv_h_f::ChebConv
    w_f
    b_f
    conv_x_c::ChebConv
    conv_h_c::ChebConv
    w_c
    b_c
    conv_x_o::ChebConv
    conv_h_o::ChebConv
    w_o
    b_o
    k::Int
    state0
    in::Int
    out::Int
end

Flux.@functor GConvLSTMCell

function GConvLSTMCell(ch::Pair{Int, Int}, k::Int, n::Int;
                        bias::Bool = true,
                        init = Flux.glorot_uniform,
                        init_state = Flux.zeros32)
    in, out = ch
    # input gate
    conv_x_i = ChebConv(in => out, k; bias, init)
    conv_h_i = ChebConv(out => out, k; bias, init)
    w_i = init(out, 1)
    b_i = bias ? Flux.create_bias(w_i, true, out) : false
    # forget gate
    conv_x_f = ChebConv(in => out, k; bias, init)
    conv_h_f = ChebConv(out => out, k; bias, init)
    w_f = init(out, 1)
    b_f = bias ? Flux.create_bias(w_f, true, out) : false
    # cell state
    conv_x_c = ChebConv(in => out, k; bias, init)
    conv_h_c = ChebConv(out => out, k; bias, init)
    w_c = init(out, 1)
    b_c = bias ? Flux.create_bias(w_c, true, out) : false
    # output gate
    conv_x_o = ChebConv(in => out, k; bias, init)
    conv_h_o = ChebConv(out => out, k; bias, init)
    w_o = init(out, 1)
    b_o = bias ? Flux.create_bias(w_o, true, out) : false
    state0 = (init_state(out, n), init_state(out, n))
    return GConvLSTMCell(conv_x_i, conv_h_i, w_i, b_i,
                         conv_x_f, conv_h_f, w_f, b_f,
                         conv_x_c, conv_h_c, w_c, b_c,
                         conv_x_o, conv_h_o, w_o, b_o,
                         k, state0, in, out)
end

function (gclstm::GConvLSTMCell)((h, c), g::GNNGraph, x)
    # input gate
    i = gclstm.conv_x_i(g, x) .+ gclstm.conv_h_i(g, h) .+ gclstm.w_i .* c .+ gclstm.b_i 
    i = Flux.sigmoid_fast(i)
    # forget gate
    f = gclstm.conv_x_f(g, x) .+ gclstm.conv_h_f(g, h) .+ gclstm.w_f .* c .+ gclstm.b_f
    f = Flux.sigmoid_fast(f)
    # cell state
    c = f .* c .+ i .* Flux.tanh_fast(gclstm.conv_x_c(g, x) .+ gclstm.conv_h_c(g, h) .+ gclstm.w_c .* c .+ gclstm.b_c)
    # output gate
    o = gclstm.conv_x_o(g, x) .+ gclstm.conv_h_o(g, h) .+ gclstm.w_o .* c .+ gclstm.b_o
    o = Flux.sigmoid_fast(o)
    h =  o .* Flux.tanh_fast(c)
    return (h,c), h
end

function Base.show(io::IO, gclstm::GConvLSTMCell)
    print(io, "GConvLSTMCell($(gclstm.in) => $(gclstm.out))")
end

GConvLSTM(ch, k, n; kwargs...) = Flux.Recur(GConvLSTMCell(ch, k, n; kwargs...))
Flux.Recur(tgcn::GConvLSTMCell) = Flux.Recur(tgcn, tgcn.state0)

(l::Flux.Recur{GConvLSTMCell})(g::GNNGraph) = GNNGraph(g, ndata = l(g, node_features(g)))
GraphNeuralNetworks._applylayer(l::Flux.Recur{GConvLSTMCell}, g::GNNGraph, x) = l(g, x)
GraphNeuralNetworks._applylayer(l::Flux.Recur{GConvLSTMCell}, g::GNNGraph) = l(g)


g = GNNGraph(SimpleDiGraph(ones(10, 10) - I(10)))
features = [rand(2, 10) for _ in 1:40]
targets = [rand(1, 10) for _ in 1:10]

model = GNNChain(GConvLSTM(2 => 10, 2, 10), Dense(10, 1))
opt = Flux.setup(Adam(0.00001), model)
accuracy(ŷ, y) = 1 - Statistics.norm(y - ŷ) / (Statistics.norm(y) + 0.01)
sqnorm(x) = sum(abs2, x)
loss_function(ŷ, y, m, λ) = Flux.huber_loss(ŷ, y) 

for epoch in 1:100
    for (x, y) in zip(features, targets)
        x, y = (x, y)
        grads = Flux.gradient(model) do model
            ŷ = model(g, x)
            loss_function(ŷ, y, model, 0.2)
        end
        Flux.update!(opt, model, grads[1])
    end

    if epoch % 10 == 0
        loss = mean([loss_function(model(g, x), y, model, 0.2)  for (x, y) in zip(features, targets)])
        println("Epoch: $epoch, Loss: $loss")
    end
end