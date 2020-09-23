%% setp00_a_preprocess
%
%Preprocess Word2Sense dataset and only keep variables intersecting with
%THINGS dataset (Hebart et al)
% 
%PATH: assumes  "THINGSdataset" and "Word2Sense" folders located one
%directory up with original datasets.
%- Word2Sense: downloaded zip file from
% https://github.com/abhishekpanigrahi1996/Word2Sense with pretrained
% senses (Panigrahi et al., 2019)
% - things dataset: http://doi.org/10.17605/osf.io/jum2f (Hebart et al.,
% 2019)
%
% Caterina Magri | Johns Hopkins University 

%Clear Workspace 
close all
clear all
clc

%Load elements
THINGS = dir('../THINGSdataset/Main/images');
THINGS = THINGS(3:end);
THINGSInfo = readtable('../THINGSdataset/Main/things_concepts.csv');


Wrd2Sns = readtable('../Word2Sense/Word2Sense.txt');

%% Intersect Word2Sense and THINGs datasets
%Only keep rows that are both in Word2Sense and in THINGs
KeepRows = zeros(1,size(Wrd2Sns,1));
whichObjects = zeros(1,size(THINGS,1));
% counter = 1;
% idcount = [];
ononym = 0;
for i = 1 : size(THINGS,1)
    if any(strcmp(Wrd2Sns.Var1 , THINGS(i).name))%if these is at elast one word matching ebtween two datasets
        
        thisIndex = find(strcmp(Wrd2Sns.Var1 , THINGS(i).name)); %track the index 
        
        if KeepRows(thisIndex) == 0  %temporary fix to avoid cases of different objects wiht same names, we just take them out
            %next we might just rename them. RIght now we just remove any
            %omonyms
            KeepRows(thisIndex) = 1;
            whichObjects(i) = 1;
        else
            ononym = ononym+1; %to keep track of how many ononyms there are
            KeepRows(thisIndex) = 0; %we remove them from the intersection
        end
       
    else
    end
end
%Just keep the rows present in both datasets
ThingsWrd2Sns = Wrd2Sns(KeepRows==1,:);
KeptTHINGSInfo = THINGSInfo(whichObjects'==1,:);


%Assign names of to variables in table
ThingsWrd2Sns.Properties.VariableNames(1) = {'item'};
for i = 2:size(ThingsWrd2Sns,2)
    ThingsWrd2Sns.Properties.VariableNames(i) = {sprintf('sense%03d',i-1)};
end

%Sort according to the label
[~,idW2S] = sort(ThingsWrd2Sns.item);
% issorted(idW2S)
[~,isTHG] = sort(KeptTHINGSInfo.Word);
% issorted(isTHG)

%make sure all words are matching and rearrange words by the same indices
[td,idx] = ismember(ThingsWrd2Sns.item,KeptTHINGSInfo.Word);
% issorted(idx)
ThingsWrd2Sns = ThingsWrd2Sns(logical(idx),:);%reindex info

[td2,idx2] = ismember(KeptTHINGSInfo.Word,ThingsWrd2Sns.item);
KeptTHINGSInfo = KeptTHINGSInfo(logical(idx2),:);

[td3,idx3] = ismember(KeptTHINGSInfo.Word,ThingsWrd2Sns.item);
% issorted(idx3)


save('ThingsSenses', 'ThingsWrd2Sns','KeptTHINGSInfo')
% Write data to text file
writetable(ThingsWrd2Sns, 'ThingsWrd2Sns.txt')
writetable(KeptTHINGSInfo, 'KeptTHINGSInfo.txt')


