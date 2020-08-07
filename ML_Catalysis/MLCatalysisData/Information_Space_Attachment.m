%==========================================================================
%This code provides visualization of the full information space in
%comparison to the currently explored space.
%==========================================================================


clear
clc
close all

load 'Catalyst_Data'


load 'Promoter_Data'


load 'Support_Data'


load 'data_table_all_points.mat'

A = table2array(data_table);

for i = 1:length(A)
    
    if A(i,1) == -5.1800
        
        A(i,1) = -5.8100;
        
    end
end

%%

%Normalize Data Points

[A1,mu,sigma] = zscore(A(:,[1:18]));

C = corrcoef(A(:,[1:18]));

[V,D] = eig(C);

[d,ind] = sort(diag(D),'descend');

Ds = D(ind,ind);

Vs = V(:,ind);

PC1a = Vs(:,1)'*A1(:,[1:18])';

PC2a = Vs(:,2)'*A1(:,[1:18])';

PC3a = Vs(:,3)'*A1(:,[1:18])';

Promoter1(29,:) = zeros(1,8);

Promoter(29,:) = (Promoter1(29,:));

Cat_Load = [10.244; -0.4781; 5];

Prom_Load = [10.5887; -0.4512; 5];


%Permutation Matrix

A = Catalyst';

B = Cat_Load';

C = Promoter';

D = Prom_Load';

E = Support';

Perm = combvec(A,B,C,D,E)';

for ii=1:size(Perm,1)
    Perm(ii,:) = (Perm(ii,:) - mu(1:18)) ./ sigma(1:18);
end

PC1 = Vs(:,1)'*Perm';

PC2 = Vs(:,2)'*Perm';

PC3 = Vs(:,3)'*Perm';

hold on 
scatter(PC1,PC2,'Fill')

scatter(PC1a,PC2a,'Fill')

set(gca,'fontsize',26)
legend('Span of Information Space','Explored Space')
box on
grid on
legend({'Span of Information Space','Currently Explored Space'},'Interpreter','latex')
xlabel('Principal Component 1','Interpreter','latex')
ylabel('Principal Component 2','Interpreter','latex')