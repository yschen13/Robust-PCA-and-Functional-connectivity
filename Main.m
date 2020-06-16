% Robust PCA project for ECE273
% Paper reference: Robust Principle Analysis? Candes, E.J., Journal of the ACM 2011
clear; close all
cd('/home/yuchen/Documents/RobustPCA')
addpath('/home/yuchen/Documents/RobustPCA/inexact_alm_rpca/')
addpath('/home/yuchen/Documents/RobustPCA/inexact_alm_rpca/PROPACK')


%% Test of different algorithms
% revised version 
[M,L,S] = RandSL(500,0.05,0.05,1);
[A,E,iter] = inexact_alm_rpca_revised(M,1/sqrt(n));
norm(L-A,'fro')/norm(L,'fro')

%% Numerical simulations: Paper Table 1

% square matrices
n_list = [500 1000 2000 3000];
error_relative = [];
for i = 1:length(n_list)
    n = n_list(i); 
    [M,L,S] = RandSL(n,0.05,0.05,1);
    [A,E,iter] = inexact_alm_rpca(M,1/sqrt(n)); % A is low-rank; E is sparse
    error_relative = [norm(L-A,'fro')/norm(L,'fro') error_relative];
end

f = figure('Position',[0 0 200 200]);
plot(n_list,error_relative,'-ko','LineWidth',1.5);
xlabel('Square matrix dimension'); ylabel('Recover error')
set(gca, 'YScale', 'log');
saveas(f,'Error_dim.png')


% PCA decomposition
n=500;
[M,L,S] = RandSL(n,0.05,0.05,0.005);
M_demean = M - repmat(mean(M),[n 1]);
L_demean = L - repmat(mean(L),[n 1]);
[COEFF, SCORE,~,~,explained,mu] = pca(M_demean);
r_chosen = 100;
L_hat = SCORE(:,1:r_chosen)*SCORE(:,1:r_chosen)'*M_demean;
norm(M_demean(:)-L_hat(:),2)
norm(M_demean-L_hat,'fro')/norm(L_demean,'fro')

[U,sigma,V] = svd(M);
L_hat = U(:,1:r_chosen) * V(:,1:r_chosen)';
norm(L-L_hat,'fro')/norm(L,'fro')


f = figure('Position',[0 0 800 600]);
subplot(2,2,1);imagesc(L);colorbar;title(['L_0: rank=' num2str(r)])
subplot(2,2,2);imagesc(S);colorbar;title(['S_0: sparsity=0.05'])
subplot(2,2,3);imagesc(L+S);colorbar;title('M')
subplot(2,2,4);imagesc(A);colorbar;title('Recovered L_0')
saveas(f,'GrossCorrupt.png')

% a low rank matrix example with pattern
n = 500; r = 2; k=0.05*n^2; % sparsity 5%
L = zeros(n); 
for i = 1:r
	index_b = (i-1)*n/r + 1;
	index_e = i*n/r;
	L(index_b:index_e,index_b:index_e) = 1/n;
end
A = rand(n);E = double(A<0.5)*2-1; % 
A = rand(n);O = double(A<0.05); % support set
S = E.* O; % sparse -1/1 matrix
M = L + S;
[A,E,iter] = inexact_alm_rpca(M,1/sqrt(n)); % A is low-rank; E is sparse
error_relative = norm(L-A,'fro')/norm(L,'fro');

f = figure('Position',[0 0 800 600]);
subplot(2,2,1);imagesc(L);colorbar;title(['L_0: rank=' num2str(r)])
subplot(2,2,2);imagesc(S);colorbar;title(['S_0: sparsity=0.05'])
subplot(2,2,3);imagesc(L+S);colorbar;title('M')
subplot(2,2,4);imagesc(A);colorbar;title('Recovered L_0')
saveas(f,'PatternCorrupted.png')

% not grossly corrupted data i.e. vary the magnitude of S
magS = 10.^(log10(1/n):0.5:1);
error_relative = [];
MessPer = 0.1;
for i = 1:length(magS)   
    n = 500; 
    [M,L,S] = RandSL(n,0.05,MessPer,magS(i));
    [A,E,iter] = inexact_alm_rpca(M,1/sqrt(n)); % A is low-rank; E is sparse
    error_relative = [norm(L-A,'fro')/norm(L,'fro') error_relative];
end

%% Transition plot
% rank and sparsity; random signal, coherent signal, matrix completion

% for random signal
n = 400;
sparsity = 0.01:0.05:0.5;
rank_per = 0.01:0.05:0.5;
error_matrix = zeros(length(sparsity),length(rank_per),10);
for i = 1:length(sparsity)
    for j = 1:length(rank_per)
        trial = 1;
        while trial < 11
            [M,L,S] = RandSL(n,rank_per(j),sparsity(i),1);
            [~,A,~,~] = evalc('inexact_alm_rpca(M,1/sqrt(n))');
            error_matrix(i,j,trial) = norm(L-A,'fro')/norm(L,'fro');
            trial = trial + 1;
        end
    end
    ['Finish ' num2str(i/length(sparsity))]
end
save('Random_error_matrix.mat','error_matrix')

Type = 'Coherent';
% Type = 'Random';
load([Type '_error_matrix_small.mat'])
thres = 10^-2;
success_rate = mean(double(error_matrix < thres),3);
f = figure('Position',[0 0 400 400]);
heatmap(rank_per,sparsity,success_rate);
xlabel('Rank/n'); ylabel('Sparsity'); title([Type ' signal: success rate'])
saveas(f,[Type '_TransitionPlot.png'])


        

%% Application: 
% Generate multiple covariance matrices with variations and find the
% low-rank part
n = 500;
k = 0.05*n;
pattern = randn([n 1]);
L = repmat(pattern,[1 n]);
A = rand(n);
E = double(A<0.5)*2-1; % random sign
A = rand(n);
O = double(A<0.05); % support set
S = E.* O; % sparse -1/1 matrix
M = L + S;
[A,E,iter] = inexact_alm_rpca(M,1/sqrt(n));
norm(L-A,'fro')/norm(L,'fro')

%{
    
 Different DCM realizations 

% This simulation has too little inter-individual variability
% decrease the sample size could create higher varibility in the cov matrices 
% data_dic = '/nadata/cnl/data/yuchen/Balloon/Manuscript/DCM_cxcx_56789/Ind_Rand/';
data_dic = '/nadata/cnl/data/yuchen/Balloon/Manuscript/DCM_cxcx_56789/Ind_LargeRand/'
i = 1;
tmp = dlmread([data_dic 'DCMV_dt0.01_' num2str(i) '.txt']);
mat = cov(tmp(1:50000,:));

f = figure;
imagesc(mat-diag(diag(mat)));colorbar
saveas(f,'ExampleConnectivity_LargeRand.png')


TS = {};
Matrix = zeros(60*60,50);
for i = 1:50
    tic
    TS{i} = dlmread([data_dic 'DCMV_dt0.01_' num2str(i) '.txt']);
    tmp = cov(TS{i});
    Matrix(:,i) = tmp(:);
    toc
end
sum(std(Matrix,[],2)) % summation of connection variance

addpath('/home/yuchen/Documents/Balloon/Manuscript/SupportFunctions/FSLNets/')
addpath('/home/yuchen/Documents/Balloon/Manuscript/SupportFunctions/L1precision/')
Matrix_icov = zeros(60*60,50);
for i = 1:50
    tic
    ts = TS{i};
    netmats = nets_netmats(ts,1,'icov',10);
    Matrix_icov(:,i) = netmats(:);
    toc
end

f = figure;
imagesc(netmats);colorbar
title('Precision matrix')
saveas(f,'icov.png')


f = figure;
for i = 1:4
    subplot(2,2,i)
    tmp = reshape(Matrix(:,i),[60 60]);
    imagesc(tmp-diag(diag(tmp)))
end
saveas(f,'ExampleConnectivity.png')


[L_hat,S_hat,~] = inexact_alm_rpca(Matrix,1/sqrt(max(size(Matrix))));
f = figure;
subplot(1,2,1)
imagesc(Matrix);
xlabel('Simulation index'); ylabel('Connectivity index')
subplot(1,2,2)
imagesc(L_hat);
title('Recovered low rank')
saveas(f,'DCM_Matrices.png')

f = figure;
for i = 1:4
    subplot(2,2,i)
    tmp = reshape(L_hat(:,i),[60 60]);
    imagesc(tmp-diag(diag(tmp)))
end
saveas(f,'ExampleLowRank_Connectivity.png')

%}


%% DCM simulate connectivity matrices with block contamination
% dVdt = GV + u
N = 50;

G = dlmread('~/Documents/Balloon/Manuscript/SimScripts/GT.txt');
G = -G(1:50,1:50)';
G = G + randn(N)*0.1;
for i = 1:50
    G(i,1:i) = 0;
end
G = G - eye(N);
[V,D] = eig(G);
lambda = sort(real(diag(D)),'descend') 
f = figure; imagesc(G);colorbar;saveas(f,'GT.png')
% G = randn(N);
% in triangular matrices, eigenvalues are just the diagonal values


NReal = 50;
sparsity = 0.05; 
A = rand([N^2 NReal]); E = double(A<0.5)*2-1; % sign of S
A = rand([N^2 NReal]); O = double(A<sparsity); % support set
S = E.* O; % sparse -1/1 matrix
L = repmat(reshape(G,[N^2 1]),[1 NReal]);
M = L + S;
[L_hat,S_hat,~] = inexact_alm_rpca(M,1/sqrt(max(size(M))));
norm(L-L_hat,'fro')/norm(L,'fro')



G_real = reshape(M(:,1),[50 50]);
for i = 1:50
    G_real(i,1:i) = 0;
end
G_real = G_real - eye(N);
[V,D] = eig(G);
lambda = sort(real(diag(D)),'descend') 



RandInput = 0;
deltaT = .01;
T = 1000/deltaT; % this nominator term is in seconds
V = zeros(T, N); V(1,:) = ones([1 N]);
I = zeros(T, N);
f = figure;imagesc(G_real);colorbar;saveas(f,'GT_corrupted.png')
tic
for t= 2:T
    W = randn(N,1)*RandInput;
    u = W';
    I(t,:) = (G_real*V(t-1, :)')';
    V(t, :) = V(t-1, :) + (I(t,:)+ u)*deltaT;
    if any(isnan(V(t,:)))
        disp('NaN occured')
        break
    end
end

mat = cov(V);
f = figure;imagesc(cov(V));colorbar;saveas(f,'cov_corrupted.png')




% HCP dataset: can we find a common network from covarinace (precision) matrix? 
load('/home/yuchen/Documents/fMRI_Real/HCP/HCP_PTN1200/Analysis/ConvMatrix_IC100.mat')


IC = load('~/Documents/fMRI_Real/HCP/HCP_PTN1200/FC-SC/GT/SelectedIC.mat');
[num,txt,raw] = xlsread('~/Documents/fMRI_Real/HCP/HCP_PTN1200/FC-SC/GT/ICmap_HCP_Hag.xlsx',4,'A:D');
% num is the selected ICs
subnet = raw(2:end,4); subnet_name = unique(subnet);
subnet_idx = zeros(60,1);
for i = 1:60
    if contains(subnet{i},subnet_name{1})
        subnet_idx(i) = 1;
    elseif contains(subnet{i},subnet_name{2})
        subnet_idx(i) = 2;
    elseif contains(subnet{i},subnet_name{3})
        subnet_idx(i) = 3;
    elseif contains(subnet{i},subnet_name{4})
        subnet_idx(i) = 4;
    elseif contains(subnet{i},subnet_name{5})
        subnet_idx(i) = 5;
    elseif contains(subnet{i},subnet_name{6})
        subnet_idx(i) = 6;
    end
end
[tmp,idx] = sort(subnet_idx);
IC_sorted = num(idx);


N = 100
M = size(cov_list,1);
Matrix = zeros(N*N,M);
for i = 1:N
    mat = cov_list{i,5};
    Matrix(:,i) = reshape(mat-diag(diag(mat)),[N*N 1]);
end
lambda = 1/sqrt(max(size(Matrix)));
[L_hat,S_hat,~] = inexact_alm_rpca(Matrix,lambda);



f = figure;
subplot(1,3,1);imagesc(Matrix);title('Original connectivity');
subplot(1,3,2);imagesc(L_hat); title('Low rank')
subplot(1,3,3);imagesc(S_hat); title('Sparse')
saveas(f,'Recovery_HCP.png')

randselect = randi(100,[1 4]);
f = figure; 
for i = 1:4
    subplot(2,2,i)
    mat = cov_list{randselect(i),5};
    mat = mat-diag(diag(mat));
    imagesc(mat(IC_sorted,IC_sorted));colorbar;
end
saveas(f,'ExampleCov_HCP.png')

f = figure;
for i = 1:4
    subplot(2,2,i)
    mat = reshape(L_hat(:,randselect(i)),[N N]);
    mat = mat-diag(diag(mat));
    imagesc(mat(IC_sorted,IC_sorted));colorbar;

end
saveas(f,'ExampleRecovery_HCP.png')

f = figure;
for i = 1:4
    subplot(2,2,i)
    mat = reshape(S_hat(:,randselect(i)),[N N]);
    mat = mat-diag(diag(mat));
    imagesc(mat(IC_sorted,IC_sorted));colorbar;
end
saveas(f,'ExampleSparseRecovery_HCP.png')

% eigenmode
[S V D] = svd(L_hat);
sv = diag(V);

f = figure;
for i = 1:4
    subplot(2,2,i)
    mat = reshape(S(:,i),[N N]);
    imagesc(mat(IC_sorted,IC_sorted));colorbar;
    title(['PC' num2str(i) ' of the low rank'])
end
saveas(f,'ExampleLowRankPC_HCP.png')

f = figure;
avgmat = reshape(mean(Matrix,2),[N N]);
imagesc(avgmat(IC_sorted,IC_sorted));colorbar;
title(['Average covaraince matrix'])
saveas(f,'Avg_cov.png')

PC1_mat = reshape(S(:,1),[N N);
Y = pdist(PC1_mat);
Z = linkage(Y)
f = figure;dendrogram(Z,0);saveas(f,'PC1_dendrogram.png')

f = figure('Position',[0 0 600 700]);
for i = 1:2
    subplot(3,2,i)
    mat = cov_list{randselect(i),5};
    mat = mat-diag(diag(mat));
    imagesc(mat(IC_sorted,IC_sorted));colorbar;
    title(['Covariance matrix: subject ' num2str(randselect(i))])
    subplot(3,2,i+2)
    mat = reshape(L_hat(:,randselect(i)),[N N]);
    mat = mat-diag(diag(mat));
    imagesc(mat(IC_sorted,IC_sorted));colorbar;
    title(['Recoved low-rank matrix: subject ' num2str(randselect(i))])
    subplot(3,2,i+4)
    mat = reshape(S_hat(:,randselect(i)),[N N]);
    mat = mat-diag(diag(mat));
    imagesc(mat(IC_sorted,IC_sorted));colorbar;
    title(['Recoved sparse matrix: subject ' num2str(randselect(i))])
end
saveas(f,'Example_HCP.png')

f = figure('Position',[0 0 700 250]);
subplot(1,2,1)
mat = reshape(S(:,1),[N N]);
imagesc(mat(IC_sorted,IC_sorted));colorbar;
title(['PC' num2str(1) ' of the low rank'])
subplot(1,2,2)
avgmat = reshape(mean(Matrix,2),[N N]);
imagesc(avgmat(IC_sorted,IC_sorted));colorbar;
title(['Average covaraince matrix'])
saveas(f, 'PC_Avg_HCP.png')
