# Copyright (c) 2025, Google Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of Google Inc. nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_dir = 'data/'

figure_dir = 'figures/'
if not os.path.exists(figure_dir):
  os.makedirs(figure_dir)

### Figure 2A

data = pd.read_csv(data_dir + 'figure_2_a_data.csv')

plt.figure()
plt.bar(data['bin'], data['count'], color='k')
plt.grid('on')
xlab = 'Pregnancy length (weeks)'
ylab = 'Participants'
plt.ylabel(ylab)
plt.xlabel(xlab)
plt.title('Figure 2a')
file_path = os.path.join(figure_dir, 'Figure 2a.png')
plt.savefig(file_path, transparent=True)


### Figure 2B

data = pd.read_csv(data_dir + 'figure_2_b_data.csv')

plt.figure()
plt.plot(data['week'], data['data'], '-ok')
plt.grid('on')
xlab = 'Time pregnant (weeks)'
ylab = 'Participants (%)'
plt.ylabel(ylab)
plt.xlabel(xlab)
plt.title('Figure 2b')
file_path = os.path.join(figure_dir, 'Figure 2b.png')
plt.savefig(file_path, transparent=True)


### Figure 2C

data = pd.read_csv(data_dir + 'figure_2_c_data.csv')

plt.figure()
plt.plot(data['week'], data['mean'], '-ok')
plt.grid('on')
xlab = 'Time (weeks)'
ylab = 'Participants (%)'
plt.ylabel(ylab)
plt.xlabel(xlab)
plt.title('Figure 2c')
file_path = os.path.join(figure_dir, 'Figure 2c.png')
plt.savefig(file_path, transparent=True)

### Figure 3

data = pd.read_csv(data_dir + 'figure_3_data.csv')

plt.figure()
plt.plot(data['day'], data['mean'], '-k')
plt.fill_between(
    data['day'],
    list(data['mean'].values - data['std'].values),
    list(data['mean'].values + data['std'].values),
    alpha=0.2,
    color='k',
)

plt.grid('on')
xlab = 'Time pregnant (days)'
ylab = 'Heart rate change (bpm)'
plt.ylabel(ylab)
plt.xlabel(xlab)
plt.title('Figure 3')
file_path = os.path.join(figure_dir, 'Figure 3.png')
plt.savefig(file_path, transparent=True)


### Figure 4A

data = pd.read_csv(data_dir + 'figure_4_a_data.csv')

plt.figure()
plt.plot(data['week'], data['mean'], '-ok')
plt.fill_between(
    data['week'],
    list(data['mean'].values - data['std'].values),
    list(data['mean'].values + data['std'].values),
    alpha=0.2,
    color='k',
)

plt.grid('on')
xlab = 'Time pregnant (weeks)'
ylab = 'Normalized TIB (min)'
plt.ylabel(ylab)
plt.xlabel(xlab)
plt.title('Figure 4a')
file_path = os.path.join(figure_dir, 'Figure 4a.png')
plt.savefig(file_path, transparent=True)


### Figure 4B

data = pd.read_csv(data_dir + 'figure_4_b_data.csv')

plt.figure()
plt.plot(data['week'], data['percentile_10'], '-k', alpha=0.2)
plt.plot(data['week'], data['percentile_25'], '-k', alpha=0.5)
plt.plot(data['week'], data['percentile_50'], '-k', alpha=0.9)
plt.plot(data['week'], data['percentile_75'], '-k', alpha=0.5)
plt.plot(data['week'], data['percentile_90'], '-k', alpha=0.2)

##uncomment the line below to add the confidence interval to plot
# plt.fill_between(data['week'],list(data['mean'].values-data['std'].values),list(data['mean'].values+data['std'].values),alpha=0.2,color='k')

plt.grid('on')
xlab = 'Time pregnant (weeks)'
ylab = 'Total TIB (min)'
plt.ylabel(ylab)
plt.xlabel(xlab)
plt.title('Figure 4b')
file_path = os.path.join(figure_dir, 'Figure 4b.png')
plt.savefig(file_path, transparent=True)

### Figure 4C

data = pd.read_csv(data_dir + 'figure_4_c_data.csv')

plt.figure()
plt.plot(data['week'], data['mean'], '-ok')
plt.fill_between(
    data['week'],
    list(data['mean'].values - data['std'].values),
    list(data['mean'].values + data['std'].values),
    alpha=0.2,
    color='k',
)

plt.grid('on')
xlab = 'Time pregnant (weeks)'
ylab = 'Participants(%)'
plt.ylabel(ylab)
plt.xlabel(xlab)
plt.title('Figure 4c')
file_path = os.path.join(figure_dir, 'Figure 4c.png')
plt.savefig(file_path, transparent=True)


### Figure 4 D

data = pd.read_csv(data_dir + 'figure_4_d_data.csv')

plt.figure()
plt.plot(data['week'], data['mean'], '-k')
plt.fill_between(
    data['week'],
    list(data['mean'].values - data['std'].values),
    list(data['mean'].values + data['std'].values),
    alpha=0.2,
    color='k',
)

plt.grid('on')
xlab = 'Time pregnant (weeks)'
ylab = 'Normalized TST (min)'
plt.ylabel(ylab)
plt.xlabel(xlab)
plt.title('Figure 4d')
file_path = os.path.join(figure_dir, 'Figure 4d.png')
plt.savefig(file_path, transparent=True)


### Figure 4 E

data = pd.read_csv(data_dir + 'figure_4_e_data.csv')
allcolor = (
    np.array([[0, 144, 181], [32, 133, 78], [255, 135, 39], [188, 60, 41]])
    / 256.0
)

plt.figure()
plt.plot(data['week'], data['deep'], color=allcolor[0], label='deep')
plt.plot(data['week'], data['light'], color=allcolor[1], label='light')
plt.plot(data['week'], data['rem'], color=allcolor[2], label='rem')
plt.plot(data['week'], data['wake'], color=allcolor[3], label='wake')

plt.grid('on')
xlab = 'Time pregnant (weeks)'
ylab = 'Normalized sleep stage (min)'
plt.ylabel(ylab)
plt.xlabel(xlab)
plt.title('Figure 4e')
file_path = os.path.join(figure_dir, 'Figure 4e.png')
plt.savefig(file_path, transparent=True)


### Figure 4 F

data = pd.read_csv(data_dir + 'figure_4_f_data.csv')

plt.figure()
plt.plot(data['week'], data['percentile_10'], '-k', alpha=0.2)
plt.plot(data['week'], data['percentile_25'], '-k', alpha=0.5)
plt.plot(data['week'], data['percentile_50'], '-k', alpha=0.9)
plt.plot(data['week'], data['percentile_75'], '-k', alpha=0.5)
plt.plot(data['week'], data['percentile_90'], '-k', alpha=0.2)

##uncomment the line below to add the confidence interval to plot
# plt.fill_between(data['week'],list(data['mean'].values-data['std'].values),list(data['mean'].values+data['std'].values),alpha=0.2,color='k')

plt.grid('on')
xlab = 'Time pregnant (weeks)'
ylab = 'Sleep efficiency (%)'
plt.ylabel(ylab)
plt.xlabel(xlab)
plt.title('Figure 4f')
file_path = os.path.join(figure_dir, 'Figure 4f.png')
plt.savefig(file_path, transparent=True)


### Figure 5 A

data = pd.read_csv(data_dir + 'figure_5_a_data.csv')

plt.figure()
plt.plot(data['week'], data['mean'], '-ok')
plt.fill_between(
    data['week'],
    list(data['mean'].values - data['std'].values),
    list(data['mean'].values + data['std'].values),
    alpha=0.2,
    color='k',
)

plt.grid('on')
xlab = 'Time (weeks)'
ylab = 'Participants (%)'
plt.ylabel(ylab)
plt.xlabel(xlab)
plt.title('Figure 5a')
file_path = os.path.join(figure_dir, 'Figure 5a.png')
plt.savefig(file_path, transparent=True)


### Figure 5 B

data = pd.read_csv(data_dir + 'figure_5_b_data.csv')

plt.figure()
plt.plot(data['week'], data['percentile_10'], '-k', alpha=0.2)
plt.plot(data['week'], data['percentile_25'], '-k', alpha=0.5)
plt.plot(data['week'], data['percentile_50'], '-k', alpha=0.9)
plt.plot(data['week'], data['percentile_75'], '-k', alpha=0.5)
plt.plot(data['week'], data['percentile_90'], '-k', alpha=0.2)

plt.grid('on')
xlab = 'Time (weeks)'
ylab = 'Total TIB (min)'
plt.ylabel(ylab)
plt.xlabel(xlab)
plt.title('Figure 5b')
file_path = os.path.join(figure_dir, 'Figure 5b.png')
plt.savefig(file_path, transparent=True)


### Figure 5 c

data = pd.read_csv(data_dir + 'figure_5_c_data.csv')

plt.figure()
plt.plot(data['week'], data['mean'], '-ok')
plt.fill_between(
    data['week'],
    list(data['mean'].values - data['std'].values),
    list(data['mean'].values + data['std'].values),
    alpha=0.2,
    color='k',
)

plt.grid('on')
xlab = 'Time (weeks)'
ylab = 'Participants (%)'
plt.ylabel(ylab)
plt.xlabel(xlab)
plt.title('Figure 5c')
file_path = os.path.join(figure_dir, 'Figure 5c.png')
plt.savefig(file_path, transparent=True)


### Figure 5 D

data = pd.read_csv(data_dir + 'figure_5_d_data.csv')

plt.figure()
plt.plot(data['week'], data['mean'], '-ok')
plt.fill_between(
    data['week'],
    list(data['mean'].values - data['std'].values),
    list(data['mean'].values + data['std'].values),
    alpha=0.2,
    color='k',
)

plt.grid('on')
xlab = 'Time (weeks)'
ylab = 'Participants (%)'
plt.ylabel(ylab)
plt.xlabel(xlab)
plt.title('Figure 5d')
file_path = os.path.join(figure_dir, 'Figure 5d.png')
plt.savefig(file_path, transparent=True)
