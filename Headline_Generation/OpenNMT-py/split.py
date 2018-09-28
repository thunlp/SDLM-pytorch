import sys
import os
txtpath = sys.argv[1]
name, ext = os.path.splitext(txtpath)
output_path = name + '_char.txt'
with open (txtpath, 'r') as f:
	with open(output_path, 'w') as g:
		lines = f.readlines()
		for line in lines:
			line = line.replace(' ','')
			for i in range(len(line)):
				str = line[i]
				if str != '\n':
					str = str + ' '
				g.write(str)