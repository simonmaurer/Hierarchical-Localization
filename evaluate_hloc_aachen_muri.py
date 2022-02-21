#!/usr/bin/env python3

import argparse

from pathlib import Path
from pprint import pformat

from hloc import extract_features, match_features, pairs_from_covisibility, pairs_from_retrieval
from hloc import colmap_from_nvm, triangulation, localize_sfm#, visualization


def main():
    print("--HLOC Aachen evaluation--")
    
    """Parse arguments for HLOC Aachen evaluation script"""
    parser = argparse.ArgumentParser(description='HLOC Aachen evaluation script')
    parser.add_argument("-d", "--dataset_dir", type=str, default='./datasets/aachen', help="Path to dataset directory")
    parser.add_argument("-o", "--output_dir", type=str, default='./outputs/aachen', help="Path to output directory")
    args = parser.parse_args()
    
    dataset = Path(args.dataset_dir)  # change this if your dataset is somewhere else
    images = dataset / 'images/images_upright/'

    outputs = Path(args.output_dir)  # where everything will be saved
    sfm_pairs = outputs / 'pairs-db-covis20.txt'  # top 20 most covisible in SIFT model
    loc_pairs = outputs / 'pairs-query-netvlad20.txt'  # top 20 retrieved by NetVLAD
    #reference_sfm = outputs / 'sfm_superpoint+superglue'  # the SfM model we will build
    #results = outputs / 'Aachen_hloc_superpoint+superglue_netvlad20.txt'  # the result file
    reference_sfm = outputs / 'sfm_muri+NN-ratio0.8'  # the SfM model we will build
    results = outputs / 'Aachen_hloc_muri+NN-ratio0.8_netvlad20.txt'  # the result file

    # list the standard configurations available
    print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
    print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')
    
    retrieval_conf = extract_features.confs['netvlad']
    #feature_conf = extract_features.confs['superpoint_aachen']
    #matcher_conf = match_features.confs['superglue']
    feature_conf = extract_features.confs['muri']
    matcher_conf = match_features.confs['NN-ratio']
    
    features = extract_features.main(feature_conf, images, outputs)
    
    colmap_from_nvm.main(dataset / '3D-models/aachen_cvpr2018_db.nvm', dataset / '3D-models/database_intrinsics.txt', dataset / 'aachen.db', outputs / 'sfm_sift')

    pairs_from_covisibility.main(outputs / 'sfm_sift', sfm_pairs, num_matched=20)
    
    sfm_matches = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)
    
    reconstruction = triangulation.main(reference_sfm, outputs / 'sfm_sift', images, sfm_pairs, features, sfm_matches)
    
    global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(global_descriptors, loc_pairs, num_matched=20, db_prefix="db", query_prefix="query")
    
    loc_matches = match_features.main(matcher_conf, loc_pairs, feature_conf['output'], outputs)
    
    #localize_sfm.main(reconstruction, dataset / 'queries/*_time_queries_with_intrinsics.txt', loc_pairs, features, loc_matches, results, covisibility_clustering=True)  # not required with SuperPoint+SuperGlue
    localize_sfm.main(reconstruction, dataset / 'queries/*_time_queries_with_intrinsics.txt', loc_pairs, features, loc_matches, results, covisibility_clustering=True)  # not required with SuperPoint+SuperGlue
    
    
if __name__ == "__main__":
    main()
    