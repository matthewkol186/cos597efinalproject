import matplotlib.pyplot as plt
import numpy as np
import csv

tsv_file = open("ai_decouple.tsv")
read_tsv = csv.reader(tsv_file, delimiter="\t")

['', 'Male', '(.1615, .1732)', '(.1340, .1426)', '0.1941', '0.2876', '0.1522']

group_to_measure = {}
for i, row in enumerate(read_tsv):
    if i == 0:
        continue
    group = row[1]
    if 'All' in group:
        group = 'All'
    if group == '':
        continue
    mi, plur, ens_err = row[2], row[3], row[4]
    mi = mi[1:-1].split(',')
    plur = plur[1:-1].split(',')
    mi = [float(num) for num in mi]
    plur = [float(num) for num in plur]
    if group not in group_to_measure.keys():
        group_to_measure[group] = [] #[[] for _ in range(3)]
    
    group_to_measure[group].append([mi, plur, float(ens_err)])

for group in group_to_measure.keys():
    measures = group_to_measure[group]
    mis = [measure[0] for measure in measures]
    plurs = [measure[1] for measure in measures]
    errs = [measure[2] for measure in measures]
    fig = plt.figure(figsize=(3.5, 5))
    for i, mi in enumerate(mis):
        if i == 0:
            c = 'C0'
            l = 'Coupled w/ prot att'
        elif i == 1:
            c = 'C1'
            l = 'Coupled'
        elif i == 2:
            c = 'C2'
            if group == 'All':
                l = 'Decoupled - Race'
            else:
                l = 'Decoupled'
        elif i == 3:
            c = 'C3'
            l = 'Decoupled - Sex'
            #plt.plot([i*.1, i*.1], mi, c)
            #continue
        plt.plot([i*.1, i*.1], mi, c=c, label=l)
    for i, plur in enumerate(plurs):
        c = 'C{}'.format(i)
        plt.plot([1.+i*.1, 1.+i*.1], plur, c=c)
    for i, err in enumerate(errs):
        c = 'C{}'.format(i)
        plt.scatter([2.+i*.1], [err], c=c)
    plt.xticks([0, 1, 2], ['MI\nEnsemble', 'Plurality\nEnsemble', 'Classification\nError'])
    plt.legend()
    plt.title(group)
    plt.tight_layout()
    plt.savefig('images/{}.png'.format(group), dpi=300)
    plt.close()
