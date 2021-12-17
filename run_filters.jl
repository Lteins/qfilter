using Printf
using NPZ
include("util.jl")
include("kmeans_filters.jl")
include("quantum_filters.jl")

log_file = open("run_filters.log", "a")

# for name in ARGS
    # target_label = parse(Int, split(name, "-")[3][end:end])
name = "test"
reps = npzread("/content/gdrive/My Drive/Resnet/Cheat/ones.np")'
n = size(reps)[2]
polluted_mark = npzread("/content/gdrive/My Drive/Resnet/Cheat/mark.np")
eps = sum(polluted_mark)
# removed = round(Int, 1.5*eps)
removed = round(Int, 1.5*3424)

    # @printf("%s: Running PCA filter\n", name)
    # reps_pca, U = pca(reps, 1)
    # pca_poison_ind = k_lowest_ind(-abs.(mean(reps_pca[1, :]) .- reps_pca[1, :]), round(Int, 1.5*eps))
    # poison_removed = sum(pca_poison_ind[end-eps+1:end])
    # clean_removed = removed - poison_removed
    # @show poison_removed, clean_removed
    # @printf(log_file, "%s-pca: %d, %d\n", name, poison_removed, clean_removed)
    # npzwrite("output/$(name)/mask-pca-target.npy", pca_poison_ind)


    # @printf("%s: Running kmeans filter\n", name)
    # kmeans_poison_ind = .! kmeans_filter2(reps, eps)
    # poison_removed = sum(kmeans_poison_ind[end-eps+1:end])
    # clean_removed = removed - poison_removed
    # @show poison_removed, clean_removed
    # @printf(log_file, "%s-kmeans: %d, %d\n", name, poison_removed, clean_removed)
    # npzwrite("output/$(name)/mask-kmeans-target.npy", kmeans_poison_ind)

@printf("Running quantum filter\n")
quantum_poison_ind = .! rcov_auto_quantum_filter(polluted_mark, reps, eps)
# quantum_poison_ind = .! rcov_quantum_filter(reps, eps, 3)
pr = 0
for i in 1:n
    if polluted_mark[i] && quantum_poison_ind[i]
        global pr += 1
    end
    quantum_poison_ind[i] = !quantum_poison_ind[i]
end

# npzwrite("/content/gdrive/My Drive/Resnet/Cheat/Retrain/select(v2).np", quantum_poison_ind)
clean_removed = removed - pr
@show pr, clean_removed, eps
# end
