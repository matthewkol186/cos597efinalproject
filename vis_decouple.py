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

#for group in group_to_measure.keys():
for group in ['All', 'Female', 'Male', 'Black', 'White']:
    measures = group_to_measure[group]
    mis = [measure[0] for measure in measures]
    plurs = [measure[1] for measure in measures]
    errs = [measure[2] for measure in measures]
    if group in ['All', 'Female', 'Black']:
        #fig = plt.figure(figsize=(3.5, 5))
        fig = plt.figure(figsize=(5.5, 2.5))
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
        linestyle='solid'
        if group == 'Male' or group == 'White':
            linestyle='dotted'
            plt.plot([.4+i*.1, .4+i*.1], mi, c=c, linestyle=linestyle)
            plt.hlines(mi, (.4+i*.1)- 0.04, (.4+i*.1) + 0.04, colors=c)
        elif group == 'Female' or group == 'Black':
            plt.plot([i*.1, i*.1], mi, c=c, linestyle=linestyle, label=l)
            plt.hlines(mi, (i*.1)- 0.04, (i*.1) + 0.04, colors=c)
        else:
            plt.plot([i*.1, i*.1], mi, c=c, label=l, linestyle=linestyle)
            plt.hlines(mi, (i*.1)- 0.04, (i*.1) + 0.04, colors=c)
    for i, plur in enumerate(plurs):
        c = 'C{}'.format(i)
        linestyle='solid'
        add = 0
        if group == 'Male' or group == 'White':
            linestyle='dotted'
            add = .4
        plt.plot([add+1.+i*.1, add+1.+i*.1], plur, c=c, linestyle=linestyle)
        plt.hlines(plur, (add+1.+i*.1)- 0.04, (add+1.+i*.1) + 0.04, colors=c)
    for i, err in enumerate(errs):
        c = 'C{}'.format(i)
        marker = 'o'
        add = 0
        if group == 'Male' or group == 'White':
            marker = 'x'
            add = .4
        plt.scatter([add+2.+i*.1], [err], c=c, marker=marker)
    if group == 'Male' or group == 'White' or group == 'All':
        if group == 'Male':
            title = 'Sex'
            plt.plot([0], [0], c='k', linestyle='solid', label='Female')
            plt.plot([0], [0], c='k', linestyle='dotted', label='Male')
        elif group == 'White':
            title = 'Race'
            plt.plot([0], [0], c='k', linestyle='solid', label='Black')
            plt.plot([0], [0], c='k', linestyle='dotted', label='White')
            plt.plot([.8, .8], [0, .45], c='k')
            plt.plot([1.8, 1.8], [0, .45], c='k')
            plt.plot([2.8, 2.8], [0, .45], c='k')
        elif group == 'All':
            title = 'All'
        plt.title(title)
        plt.xticks([0.35, 1.35, 2.35], ['MI\nEnsemble', 'Plurality\nEnsemble', 'Classification\nError'])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('images/{}.png'.format(title), dpi=300)
        plt.close()
