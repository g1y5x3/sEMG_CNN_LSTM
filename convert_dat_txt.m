load('matching_subjects.mat')

%%
channel1 = [];
channel2 = [];
channel3 = [];
channel4 = [];
for i = 1 : 1
    channel1 = vertcat(channel1,DATA_DYSFUNC{i,1}{1,1});
    channel2 = vertcat(channel2,DATA_DYSFUNC{i,1}{1,2});
    channel3 = vertcat(channel3,DATA_DYSFUNC{i,1}{1,3});
    channel4 = vertcat(channel4,DATA_DYSFUNC{i,1}{1,4});    
end

for i = 1 : 1
    channel1 = vertcat(channel1,DATA_HEALTHY{i,1}{1,1});
    channel2 = vertcat(channel2,DATA_HEALTHY{i,1}{1,2});
    channel3 = vertcat(channel3,DATA_HEALTHY{i,1}{1,3});
    channel4 = vertcat(channel4,DATA_HEALTHY{i,1}{1,4});    
end

label = vertcat(ones(size(DATA_DYSFUNC{i,1}{1,1},1),1),...
                2*ones(size(DATA_HEALTHY{i,1}{1,1},1),1));
            
%%
for i = 1:9
    LABEL{i} = ones(size(DATA{i}{1},1),1);
end

%%
for i = 10:18
    LABEL{i} = 2 * ones(size(DATA{i}{1},1),1);
end

%% 
sum = 0;
for i = 1 : 18
    sum = sum + size(LABEL{i},1);
end
