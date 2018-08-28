
%load x and Y
load('train.mat','y')
load('train.mat','X')
%transpose y into yt
self_yt=y';
self_Xt=X';
n = size(yt,1);
%initialize data
for i=1:size(self_yt,1)
    if self_yt(i,:)==0
        self_yt(i,:)=-1;
    end
end 
%train the model
self_train_model=svmtrain(self_yt,X);

%load test data
load('test.mat','X')
load('test.mat','y')
self_testx=X;
self_testy=y';
%set lable 0 into -1
for i=1:size(self_testy,1)
    if self_testy(i,:)==0
        self_testy(i,:)=-1;
    end
end
%calculate accuracy by libsvm
[acc,label,value]=svmpredict(self_testy,self_testx,self_train_model);
%sv_coef is the alpha*y
self_alpha=self_train_model.sv_coef;
%get the filtered key x from X
self_x_ind=self_train_model.sv_indices;
getx=self_Xt(:,self_x_ind);
%calculate w from alpha*x*y
self_w=getx*self_alpha;
