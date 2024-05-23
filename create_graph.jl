using NPZ
atlas = npzread("atlas.npy")

matrix = zeros(100,100)

for i in 1:size(atlas,1)-1
    for j in 1:size(atlas,2)-1
        for k in 1:size(atlas,3)-1
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

function threshold(x,t)
    if x >= t
        return 1
    else
        return 0
    end
end
    
            
matrix = threshold.(matrix, 1)

npzwrite("graph.npy", matrix)

               