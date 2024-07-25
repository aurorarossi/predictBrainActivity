using NPZ

function create_graph_near_regions(atlas)
    matrix = zeros(Int, 100, 100)

    for i in 1:size(atlas, 1) - 1
        for j in 1:size(atlas, 2) - 1
            for k in 1:size(atlas, 3) - 1
                if atlas[i,j,k] != atlas[i+1,j,k] && atlas[i,j,k] != 0 && atlas[i+1,j,k] != 0
                    matrix[Int(atlas[i,j,k]),Int(atlas[i+1,j,k])] += 1
                end
                if atlas[i,j,k] != atlas[i,j+1,k] && atlas[i,j,k] != 0 && atlas[i,j+1,k] != 0
                    matrix[Int(atlas[i,j,k]),Int(atlas[i,j+1,k])] += 1
                end
                if atlas[i,j,k] != atlas[i,j,k+1] && atlas[i,j,k] != 0 && atlas[i,j,k+1] != 0
                    matrix[Int(atlas[i,j,k]),Int(atlas[i,j,k+1])] += 1
                end
            end
        end
    end
    return matrix
end

threshold(x, t) = (x >= t) ? 1 : 0

# Read Schaefer atlas 100 regions
atlas = npzread("data/atlas.npy")

matrix = create_graph_near_regions(atlas)       
#matrix = threshold.(matrix, 1)

npzwrite("data/graph.npy", matrix)

               