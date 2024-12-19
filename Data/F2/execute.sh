python main.py --input-file $BASE_OUTPUT_DIRECTORY/F2/f2.1-m2-targeted-boundary.jsonc
python main.py --input-file $BASE_OUTPUT_DIRECTORY/F2/f2.1-sb-targeted-boundary.jsonc

python main.py --input-file $BASE_OUTPUT_DIRECTORY/F2/f2.2-m2-targeted-clustering.jsonc
python main.py --input-file $BASE_OUTPUT_DIRECTORY/F2/f2.2-sb-targeted-clustering.jsonc


# we now export data from plots
# boundary
## m2
# - distance.boundary(norm=max)         -> 6 index 0
# - distance.clustering(norm=max)       -> 6 index 1
# - neigh.kdist(k=5, l=2, norm=div)     -> 6 index 0
## sb
# - distance.boundary(norm=max)         -> 6 index 0
# - distance.clustering(norm=max)       -> 6 index 1
# - neigh.kdist(k=50, l=2, norm=div)    -> 6 index 0

## clustering
## m2
# - distance.boundary(norm=max)         -> 6 index 0
# - distance.clustering(norm=div)       -> 6 index 0
# - neigh.kdist(k=5, l=2, norm=div)     -> 6 index 0
## sb
# - distance.boundary(norm=max)         -> 6 index 0
# - distance.clustering(norm=div)       -> 6 index 0
# - neigh.kdist(k=50, l=2, norm=div)    -> 6 index 0

# attack boundary
python -m post.re_iop \
    load \
    --input-file $BASE_OUTPUT_DIRECTORY/F2/F2.1/Output/M2-Targeted-Boundary/Output/Plots/Distance.Boundary_max/6.0_0.0_distance.html \
    --output-file-prefix $BASE_OUTPUT_DIRECTORY/F2/F2.1/Additional/m2_boundary_dist.bound_max_eps6.csv \
    --index 0

python -m post.re_iop \
    load \
    --input-file $BASE_OUTPUT_DIRECTORY/F2/F2.1/Output/SB-Targeted-Boundary/Output/Plots/Distance.Boundary_max/6.0_0.0_distance.html \
    --output-file-prefix $BASE_OUTPUT_DIRECTORY/F2/F2.1/Additional/sb_boundary_dist.bound_max_eps6.csv \
    --index 0

# attack clustering
python -m post.re_iop \
    load \
    --input-file $BASE_OUTPUT_DIRECTORY/F2/F2.2/Output/M2-Targeted-Clustering/Output/Plots/Distance.Boundary_max/6.0_0.0_distance.html \
    --output-file-prefix $BASE_OUTPUT_DIRECTORY/F2/F2.2/Additional/m2_clustering_dist.bound_max_eps6.csv \
    --index 0

python -m post.re_iop \
    load \
    --input-file $BASE_OUTPUT_DIRECTORY/F2/F2.2/Output/SB-Targeted-Clustering/Output/Plots/Distance.Boundary_max/6.0_0.0_distance.html \
    --output-file-prefix $BASE_OUTPUT_DIRECTORY/F2/F2.2/Additional/sb_clustering_dist.bound_max_eps6.csv \
    --index 0