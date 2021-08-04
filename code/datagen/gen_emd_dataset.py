from graph_data_emd import GraphDataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, help="Directory to process data", required=False,
                        default='/energyflowvol/datasets/')
    parser.add_argument("--lhco-back", action="store_true", help="Start from tail end of lhco data to get unused dataset", required=False)
    parser.add_argument("--n-jets", type=int, help="number of jets", required=False, default=100)
    parser.add_argument("--n-events-merge", type=int, help="number of events to merge", required=False, default=1)
    args = parser.parse_args()

    os.makedirs(args.input_dir,exist_ok=True)

    # log arguments
    import logging
    logging.basicConfig(filename=osp.join(args.input_dir, "logs.log"), filemode='w', level=logging.DEBUG, format='%(asctime)s | %(levelname)s: %(message)s')
    for arg, value in sorted(vars(args).items()):
            logging.info("Argument %s: %r", arg, value)

    gdata = GraphDataset(root=args.input_dir, n_jets=args.n_jets, n_events_merge=args.n_events_merge, lhco_back=args.lhco_back)