T=readtable('walk256.csv');
t=T{:,1};
a=T{:, [2:4]};
save('walk256.mat', 't', 'a');