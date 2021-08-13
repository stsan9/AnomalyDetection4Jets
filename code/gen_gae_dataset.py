from datagen.graph_data_gae import GraphDataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset path", required=True)
    parser.add_argument("--n-proc", type=int, default=1, help="number of concurrent processes")
    parser.add_argument("--n-events", type=int, default=-1, help="number of events (-1 means all)")
    parser.add_argument("--n-particles", type=int, default=-1, help="max number of particles per jet with zero-padding (-1 means all)")
    parser.add_argument("--bb", type=int, default=0, help="black box number (0 is background, -1 is the mixed rnd set)")
    parser.add_argument("--n-events-merge", type=int, default=100, help="number of events to merge")
    parser.add_argument("--part_type", choices=['xyz','relptetaphi'], help="Generate (px,py,pz) or relative (pt,eta,phi)", required=True)
    args = parser.parse_args()

    gdata = GraphDataset(root=args.dataset, bb=args.bb, n_proc=args.n_proc,
                         n_events=args.n_events, n_particles=args.n_particles,
                         n_events_merge=args.n_events_merge, part_type=args.part_type)
