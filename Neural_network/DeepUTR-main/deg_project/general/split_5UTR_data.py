import sys
import os
# Get the path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up two levels from the current directory
two_levels_up = os.path.abspath(os.path.join(current_dir, "..", ".."))

# Add the parent directory to the path
sys.path.insert(0, two_levels_up)

from deg_project.general import general_utilies
import networkx as nx
import csv
def disjoint_split(seq_path: str, output_path):
    seq_list = general_utilies.pd.read_csv(seq_path)["seq"].values.tolist()
    hash_map = {}
    k=20  # k-mer length
    l=110  # seqence fixed length
    G = nx.Graph()
    G.add_nodes_from(range(len(seq_list)))

    # passing all k-mer in the seqences and create the edges accordingly
    for i,seq in enumerate(seq_list):
        for j in range(l-k+1):
            kmer=seq[j:j+k]
            if kmer in hash_map:
                hash_map[kmer].add(i)
                edges_list = [(i,m) for m in hash_map[kmer]]
                G.add_edges_from(edges_list)
            else:
                hash_map[kmer] = {i}

    #create the connected components
    connected_components_list = list(nx.connected_components(G))
    connected_components_list.sort(key=len, reverse=True)

    #create the disjoint  train, validation, and test sets
    # This may not deliver the same ids as we provided, since we could not reproduce the random pick
    i = 0
    train = []
    while ((len(train)+len(connected_components_list[i]))<=22000): #the
        train = train+list(connected_components_list[i])
        i+=1
    test = []
    while ((len(test)+len(connected_components_list[i]))<=4000):
        test = test + list(connected_components_list[i])
        i+=1
    validation = []
    for connected_component in connected_components_list[i:]:
        validation = validation + list(connected_component)

    #save the ids in a csv file
    output_csv = [[id] for id in train]
    for i in range(len(validation)):
        output_csv[i].append(validation[i])
    for i in range(len(test)):
        output_csv[i].append(test[i])
    output_csv = [["train_ids", "validation_ids", "test_ids"]] + output_csv
    with open(output_path+'split_to_train_validation_test_disjoint_sets_ids.csv', 'w',
              newline='') as myfile:
        # wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr = csv.writer(myfile)
        wr.writerows(output_csv)

if __name__ == '__main__':
    disjoint_split("../../files/5utr_dataset/mRNA_sequences_for_5utr.csv",
                   "../../files/5utr_dataset/splitted_data/split_indices")