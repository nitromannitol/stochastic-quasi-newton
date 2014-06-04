spam_data = importdata('spam.data');
[m n] = size(spam_data);


spam_data = spam_data(1:floor(m/2), :);

cvx_begin
variable w(n-1)
expression logloss
for i = 1:size(spam_data,1)
    dataSample = spam_data(i,:);
    x = dataSample(1:n-1);
    label = dataSample(n);
    z = w'*x';
    sigmoid = 1/(1 + exp(-z));
    loss = label*log(sigmoid) + (1.0-label)*log((exp(-z)/(1+exp(-z))));
    logloss = logloss + loss;
end

minimize(-logloss)
cvx_end





%cvx_end
  %  var w(m-1)
	%label = dataSample[0].item()
	%x = dataSample[1:]
	%z = (numpy.transpose(w)*x).item()
	%if(z > 600):
%		z = 600
%	if(z < -600):
%		z = -600
%	sigmoid = 1/(1 + math.exp(-z))
%	return label*math.log(sigmoid + continuity_correction) + (1.0-label)*math.log(1.0 - sigmoid + continuity_correction)
