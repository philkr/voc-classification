from __future__ import print_function

from argparse import ArgumentParser
from glob import glob
from os import path
import numpy as np

parser = ArgumentParser()
parser.add_argument('experiment_dir', nargs='+')
args = parser.parse_args()

files = sum([glob(d+'*') for d in args.experiment_dir], [])
file_names = [path.basename(s) for s in files]

file_parts = [s.split('_') for s in file_names]
s_files = sorted(file_parts)
r_files = [s[::-1] for s in sorted([s[::-1] for s in file_parts])]

prefix, suffix = set(), set()
for i in range(len(files)-1):
	a, b = s_files[i:i+2]
	for k in range(min(len(a),len(b))):
		if a[k] != b[k]:
			break
	if k>0:
		prefix.add('_'.join(a[:k]))
	
	a, b = r_files[i:i+2]
	for k in range(min(len(a),len(b))):
		if a[-k-1] != b[-k-1]:
			break
	if k>0:
		suffix.add('_'.join(a[-k:]))

prefix, suffix = list(prefix), list(suffix)

import re
r = re.compile('(\d\.\d+) +'+' '.join(['(\d\.\d+)']*20))
file_values = {}
for n,f in zip(file_names, files):
	value = []
	for l in open(f, 'r'):
		m = r.match(l.strip())
		if m:
			value.append(m.group(1))
	if value:
		file_values[n] = float(value[-2])

score = np.zeros((len(prefix), len(suffix)))
for i,p in enumerate(prefix):
	for j,s in enumerate(suffix):
		if '%s_%s'%(p,s) in file_values:
			score[i,j] += 1

# TODO: Figure out how to filter out only valid pre and suffixes
prefix = sorted([p for i,p in enumerate(prefix) if np.sum(score[i,:]) > 0])
suffix = sorted([s for j,s in enumerate(suffix) if np.sum(score[:,j]) > 0])

print( ' '*20+' & '+' & '.join(['%10s'%p for p in prefix])+' \\\\' )
for s in suffix:
	print( '%-20s'%s+' & '+' & '.join(['%10.3f'%file_values['%s_%s'%(p,s)] if '%s_%s'%(p,s) in file_values else ' '*6+'--  ' for p in prefix])+' \\\\' )


