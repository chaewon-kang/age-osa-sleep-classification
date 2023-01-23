%% feature extraction script
%
% this script is comprised of
% 1) reading EDF and annotation files
% 2) arrange signals in 30 seconds epochs by stage
% 3) preprocess and extract features
%
% chaewon kang, sora an, 2021

clear all;
% read dataset file
T=readtable('/user/chae2089/CWpaper/SDP.xlsx','Sheet','Sheet1');
fs=200;

% SEVERE OSA[1] = 17,19,23,24,27,28,30,31,34,36,40,43,46,48,52,57,60,66,69,70,71,73,74,75,76,78,79,81,84,85,86,97,109,117,128,129,131,132,134,135,139,152,156,162
% MODERATE OSA [2] = 3,8,11,21,38,49,51,54,58,67,68,72,80,82,88,91,94,95,98,101,102,103,107,114,115,121,127,136,138,144
% MILD OSA [3] = 2,4,5,6,7,16,25,47,50,56,59,61,63,92,93,96,99,100,111,116,118,122,125,130,142,143,145
% Control[0] = 1,10,12,13,14,15,18,20,26,29,35,41,42,44,45,53,55,62,64,65,77,83,89,90,104,105,106,108,110,112,119,126,133,137,140,151,153,154,160,161,163

for n_p = 1:163
    % read edf and annotation of each subjects
	file_name=T.Var9{n_p};
    file_name2=sprintf('/user/chae2089/CWpaper/%s',file_name);
    [EDF_header, EDF_recorder] = edfread(sprintf('/user/chae2089/CWpaper/eegdata/%s.EDF',file_name)); 
    STAGE_recorder = readtable(sprintf('/user/chae2089/CWpaper/eegdata/%s_Annotation.csv',file_name));
        
    % rearrange them into 30 seconds epoch
    C = table2cell(EDF_header);
    sig = {};
    nn = 1;

    for k = 1:6
        for n=1:5:size(C,1)
            tmp = [];
            for m = n:n+4
                if m>size(C,1)
                    break
                else
                    tmp = [tmp; C{m,k}]; 
                end
            end
            sig{nn,k} = tmp;
            nn = nn+1;
        end
        nn = 1;
    end
    
    sig2=cell(size(sig,1),1);
    for ep=1:size(sig,1)
        for ch=1:6
            sig2{ep,1}=[sig2{ep,1};sig{ep,ch}'];
        end
    end

    % organize epochs into stages and their durations
    onset_epoch = STAGE_recorder.Var1;
    stages = STAGE_recorder.Var3;

    chop_onset_epoch = []; chop_stages = [];
    for n=1:length(onset_epoch)
        if contains(stages{n,1}, "Stage")==1 && contains(stages{n,1}, "No Stage")==0
            chop_onset_epoch = [chop_onset_epoch; onset_epoch(n,1)];
            chop_stages = [chop_stages; stages(n,1)];
        end
    end

    onset_epoch = chop_onset_epoch;
    stages = chop_stages;


    stages_idx = [];

    for n=1:length(stages)
        if onset_epoch(n,1)==onset_epoch(end)           
            onset_time = onset_epoch(n,1);
            durations = 1;
            stages_idx = [stages_idx; stages(n,1), onset_time, durations];
        else
            onset_time = onset_epoch(n,1);
            durations = onset_epoch(n+1)-onset_time;
            stages_idx = [stages_idx; stages(n,1), onset_time, durations];
        end
    end


    % put together epochs by stages
    stage_W = []; stage_N1 = []; stage_N2 = []; stage_N3 = []; stage_R = [];
    n_W = 1; n_N1=1; n_N2=1; n_N3=1; n_R=1;

    for k=1:length(stages_idx)
        Index = stages_idx{k,2};
        Index_dur = stages_idx{k,3};

        if strcmp(stages_idx{k,1}, "Stage - W")==1
            for m = Index : Index+(Index_dur-1)
				if size(sig2{m,1},2) == 6000
					stage_W(:,:,n_W) = sig2{m,1};
					n_W = n_W+1;
				end
            end
        elseif strcmp(stages_idx{k,1}, "Stage - N1")==1
            for m = Index : Index+(Index_dur-1)
                stage_N1(:,:,n_N1) = sig2{m,1};
                n_N1 = n_N1+1;
            end
        elseif strcmp(stages_idx{k,1}, "Stage - N2")==1
            for m = Index : Index+(Index_dur-1)
                stage_N2(:,:,n_N2) = sig2{m,1};
                n_N2 = n_N2+1;
            end
        elseif strcmp(stages_idx{k,1}, "Stage - N3")==1
            for m = Index : Index+(Index_dur-1)
                stage_N3(:,:,n_N3) = sig2{m,1};
                n_N3 = n_N3+1;
            end
        elseif strcmp(stages_idx{k,1}, "Stage - R")==1
            for m = Index : Index+(Index_dur-1)
                stage_R(:,:,n_R) = sig2{m,1};
                n_R = n_R+1;
            end
        end

    end

    % subjects with no "N3" stage
    if isempty(stage_N3)==1
        stage_W = reshape(stage_W,6,6000*size(stage_W,3));
        stage_N1 = reshape(stage_N1,6,6000*size(stage_N1,3));
        stage_N2 = reshape(stage_N2,6,6000*size(stage_N2,3));
        stage_R = reshape(stage_R,6,6000*size(stage_R,3));
        save(sprintf('/user/chae2089/CWpaper/stage_data/subject%d_W.mat',n_p), 'stage_W');
        save(sprintf('/user/chae2089/CWpaper/stage_data/subject%d_N1.mat',n_p), 'stage_N1');
        save(sprintf('/user/chae2089/CWpaper/stage_data/subject%d_N2.mat',n_p), 'stage_N2');
        save(sprintf('/user/chae2089/CWpaper/stage_data/subject%d_R.mat',n_p), 'stage_R');
        
        % preprocessing - filtering 0.5-50H via EEGLAB
        clear EEG;
        EEG.etc.eeglabvers = '2022.0'; % this tracks which version of EEGLAB is being used, you may ignore it
        EEG = pop_importdata('dataformat','matlab','nbchan',0,'data',sprintf('/user/chae2089/CWpaper/stage_data/subject%d_W.mat',n_p),'srate',200,'pnts',0,'xmin',0);
        EEG.setname='W';
        EEG = eeg_checkset( EEG );
        EEG = pop_eegfiltnew(EEG, 'locutoff',0.5);
        EEG = eeg_checkset( EEG );
        EEG = pop_eegfiltnew(EEG, 'hicutoff',50);
        EEG = eeg_checkset( EEG );
        W=EEG.data;
        
        clear EEG;
        EEG.etc.eeglabvers = '2022.0'; % this tracks which version of EEGLAB is being used, you may ignore it
        EEG = pop_importdata('dataformat','matlab','nbchan',0,'data',sprintf('/user/chae2089/CWpaper/stage_data/subject%d_R.mat',n_p),'srate',200,'pnts',0,'xmin',0);
        EEG.setname='R';
        EEG = eeg_checkset( EEG );
        EEG = pop_eegfiltnew(EEG, 'locutoff',0.5);
        EEG = eeg_checkset( EEG );
        EEG = pop_eegfiltnew(EEG, 'hicutoff',50);
        EEG = eeg_checkset( EEG );
        R=EEG.data;
        
        clear EEG;
        EEG.etc.eeglabvers = '2022.0'; % this tracks which version of EEGLAB is being used, you may ignore it
        EEG = pop_importdata('dataformat','matlab','nbchan',0,'data',sprintf('/user/chae2089/CWpaper/stage_data/subject%d_N1.mat',n_p),'srate',200,'pnts',0,'xmin',0);
        EEG.setname='N1';
        EEG = eeg_checkset( EEG );
        EEG = pop_eegfiltnew(EEG, 'locutoff',0.5);
        EEG = eeg_checkset( EEG );
        EEG = pop_eegfiltnew(EEG, 'hicutoff',50);
        EEG = eeg_checkset( EEG );
        N1=EEG.data;

        clear EEG;
        EEG.etc.eeglabvers = '2022.0'; % this tracks which version of EEGLAB is being used, you may ignore it
        EEG = pop_importdata('dataformat','matlab','nbchan',0,'data',sprintf('/user/chae2089/CWpaper/stage_data/subject%d_N2.mat',n_p),'srate',200,'pnts',0,'xmin',0);
        EEG.setname='N2';
        EEG = eeg_checkset( EEG );
        EEG = pop_eegfiltnew(EEG, 'locutoff',0.5);
        EEG = eeg_checkset( EEG );
        EEG = pop_eegfiltnew(EEG, 'hicutoff',50);
        EEG = eeg_checkset( EEG );
        N2=EEG.data;
        
        W_delta_l2=[]; W_delta_K1_2=[]; W_delta_K2_2=[]; W_theta2=[]; W_alpha2=[]; W_spin1_2=[]; W_spin2_2=[]; W_beta2=[]; 
        R_delta_l2=[]; R_delta_K1_2=[]; R_delta_K2_2=[]; R_theta2=[]; R_alpha2=[]; R_spin1_2=[]; R_spin2_2=[]; R_beta2=[]; 
        N1_delta_l2=[]; N1_delta_K1_2=[]; N1_delta_K2_2=[]; N1_theta2=[]; N1_alpha2=[]; N1_spin1_2=[]; N1_spin2_2=[]; N1_beta2=[]; 
        N2_delta_l2=[]; N2_delta_K1_2=[]; N2_delta_K2_2=[]; N2_theta2=[]; N2_alpha2=[]; N2_spin1_2=[]; N2_spin2_2=[]; N2_beta2=[]; 
    
        % PSD feature extraction of 6 channels
        for ch=1:6
			%if ch==1 || ch==2 || ch==5 || ch==6
				
            [S_w,F_w,T_w,P_w] = spectrogram(W(ch,:), kaiser(200*1,0.5),100,2^10,200,'yaxis'); P_w=P_w./sum(P_w(4:155,:),1);
            W_delta_l=[]; W_delta_K1=[]; W_delta_K2=[]; W_theta=[]; W_alpha=[]; W_spin1=[]; W_spin2=[]; W_beta=[]; 

            for t1=1:60:size(T_w,2)-31
                W_K_comp=sort(sum(P_w(7:21,t1:t1+58))); W_spindle=sort(sum(P_w(63:77,t1:t1+58)));
                W_delta_l=[W_delta_l,mean(sum(P_w(4:11,t1:t1+58)))]; W_delta_K1=[W_delta_K1,W_K_comp(59)]; W_delta_K2=[W_delta_K2,mean(W_K_comp(1:58))]; 
                W_theta=[W_theta,mean(sum(P_w(22:41,t1:t1+58)))]; W_alpha=[W_alpha,mean(sum(P_w(42:62,t1:t1+58)))]; 
                W_spin1=[W_spin1,W_spindle(59)]; W_spin2=[W_spin2,mean(W_spindle(1:58))]; W_beta=[W_beta,mean(sum(P_w(78:155,t1:t1+58)))]; 
            end

            [S_r,F_r,T_r,P_r] = spectrogram(R(ch,:), kaiser(200*1,0.5),100,2^10,200,'yaxis'); P_r=P_r./sum(P_r(4:155,:),1);
            R_delta_l=[]; R_delta_K1=[]; R_delta_K2=[]; R_theta=[]; R_alpha=[]; R_spin1=[]; R_spin2=[]; R_beta=[]; 

            for t1=1:60:size(T_r,2)-31
                R_K_comp=sort(sum(P_r(7:21,t1:t1+58))); R_spindle=sort(sum(P_r(63:77,t1:t1+58)));
                R_delta_l=[R_delta_l,mean(sum(P_r(4:11,t1:t1+58)))]; R_delta_K1=[R_delta_K1,R_K_comp(59)]; R_delta_K2=[R_delta_K2,mean(R_K_comp(1:58))]; 
                R_theta=[R_theta,mean(sum(P_r(22:41,t1:t1+58)))]; R_alpha=[R_alpha,mean(sum(P_r(42:62,t1:t1+58)))]; 
                R_spin1=[R_spin1,R_spindle(59)]; R_spin2=[R_spin2,mean(R_spindle(1:58))]; R_beta=[R_beta,mean(sum(P_r(78:155,t1:t1+58)))]; 
            end       

            [S_n1,F_n1,T_n1,P_n1] = spectrogram(N1(ch,:), kaiser(200*1,0.5),100,2^10,200,'yaxis'); P_n1=P_n1./sum(P_n1(4:155,:),1);
            N1_delta_l=[]; N1_delta_K1=[]; N1_delta_K2=[]; N1_theta=[]; N1_alpha=[]; N1_spin1=[]; N1_spin2=[]; N1_beta=[]; 

            for t1=1:60:size(T_n1,2)-31
                N1_K_comp=sort(sum(P_n1(7:21,t1:t1+58))); N1_spindle=sort(sum(P_n1(63:77,t1:t1+58)));
                N1_delta_l=[N1_delta_l,mean(sum(P_n1(4:11,t1:t1+58)))]; N1_delta_K1=[N1_delta_K1,N1_K_comp(59)]; N1_delta_K2=[N1_delta_K2,mean(N1_K_comp(1:58))]; 
                N1_theta=[N1_theta,mean(sum(P_n1(22:41,t1:t1+58)))]; N1_alpha=[N1_alpha,mean(sum(P_n1(42:62,t1:t1+58)))]; 
                N1_spin1=[N1_spin1,N1_spindle(59)]; N1_spin2=[N1_spin2,mean(N1_spindle(1:58))]; N1_beta=[N1_beta,mean(sum(P_n1(78:155,t1:t1+58)))]; 
            end

            [S_n2,F_n2,T_n2,P_n2] = spectrogram(N2(ch,:), kaiser(200*1,0.5),100,2^10,200,'yaxis'); P_n2=P_n2./sum(P_n2(4:155,:),1);
            N2_delta_l=[]; N2_delta_K1=[]; N2_delta_K2=[]; N2_theta=[]; N2_alpha=[]; N2_spin1=[]; N2_spin2=[]; N2_beta=[]; 

            for t1=1:60:size(T_n2,2)-31
                N2_K_comp=sort(sum(P_n2(7:21,t1:t1+58))); N2_spindle=sort(sum(P_n2(63:77,t1:t1+58)));
                N2_delta_l=[N2_delta_l,mean(sum(P_n2(4:11,t1:t1+58)))]; N2_delta_K1=[N2_delta_K1,N2_K_comp(59)]; N2_delta_K2=[N2_delta_K2,mean(N2_K_comp(1:58))]; 
                N2_theta=[N2_theta,mean(sum(P_n2(22:41,t1:t1+58)))]; N2_alpha=[N2_alpha,mean(sum(P_n2(42:62,t1:t1+58)))]; 
                N2_spin1=[N2_spin1,N2_spindle(59)]; N2_spin2=[N2_spin2,mean(N2_spindle(1:58))]; N2_beta=[N2_beta,mean(sum(P_n2(78:155,t1:t1+58)))]; 
            end
            
            % cumulate features
            W_delta_l2=[W_delta_l2; W_delta_l]; W_delta_K1_2=[W_delta_K1_2; W_delta_K1]; W_delta_K2_2=[W_delta_K2_2; W_delta_K2]; W_theta2=[W_theta2; W_theta]; 
            W_alpha2=[W_alpha2; W_alpha]; W_spin1_2=[W_spin1_2; W_spin1]; W_spin2_2=[W_spin2_2; W_spin2]; W_beta2=[W_beta2; W_beta]; 
            
            R_delta_l2=[R_delta_l2; R_delta_l]; R_delta_K1_2=[R_delta_K1_2; R_delta_K1]; R_delta_K2_2=[R_delta_K2_2; R_delta_K2]; R_theta2=[R_theta2; R_theta]; 
            R_alpha2=[R_alpha2; R_alpha]; R_spin1_2=[R_spin1_2; R_spin1]; R_spin2_2=[R_spin2_2; R_spin2]; R_beta2=[R_beta2; R_beta]; 
            
            N1_delta_l2=[N1_delta_l2; N1_delta_l]; N1_delta_K1_2=[N1_delta_K1_2; N1_delta_K1]; N1_delta_K2_2=[N1_delta_K2_2; N1_delta_K2]; N1_theta2=[N1_theta2; N1_theta]; 
            N1_alpha2=[N1_alpha2; N1_alpha]; N1_spin1_2=[N1_spin1_2; N1_spin1]; N1_spin2_2=[N1_spin2_2; N1_spin2]; N1_beta2=[N1_beta2; N1_beta]; 
            
            N2_delta_l2=[N2_delta_l2; N2_delta_l]; N2_delta_K1_2=[N2_delta_K1_2; N2_delta_K1]; N2_delta_K2_2=[N2_delta_K2_2; N2_delta_K2]; N2_theta2=[N2_theta2; N2_theta]; 
            N2_alpha2=[N2_alpha2; N2_alpha]; N2_spin1_2=[N2_spin1_2; N2_spin1]; N2_spin2_2=[N2_spin2_2; N2_spin2]; N2_beta2=[N2_beta2; N2_beta]; 
	
            
        end
        
        TOTAL_delta_l = [W_delta_l2, N1_delta_l2, N2_delta_l2, R_delta_l2];
        TOTAL_delta_K1 = [W_delta_K1_2, N1_delta_K1_2, N2_delta_K1_2, R_delta_K1_2];
        TOTAL_delta_K2 = [W_delta_K2_2, N1_delta_K2_2, N2_delta_K2_2, R_delta_K2_2];
        TOTAL_theta = [W_theta2, N1_theta2, N2_theta2, R_theta2];
        TOTAL_alpha = [W_alpha2, N1_alpha2, N2_alpha2, R_alpha2];
        TOTAL_spin1 = [W_spin1_2, N1_spin1_2, N2_spin1_2, R_spin1_2];
        TOTAL_spin2 = [W_spin2_2, N1_spin2_2, N2_spin2_2, R_spin2_2];
        TOTAL_beta = [W_beta2, N1_beta2, N2_beta2, R_beta2];
        %TOTAL_gamma = [W_spindle2, N1_spindle2, N2_spindle2, R_spindle2];

        % save features and their labels
        TOTAL=[TOTAL_delta_l; TOTAL_delta_K1; TOTAL_delta_K2; TOTAL_theta; TOTAL_alpha; TOTAL_spin1; TOTAL_spin2; TOTAL_beta];
        label_TOTAL =[ones(1,size(W_delta_l2,2)), 2*ones(1,size(N1_delta_l2,2)), 3*ones(1,size(N2_delta_l2,2)), 5*ones(1,size(R_delta_l2,2))]; 
        
        %figure(12); subplot(2,1,1); imagesc(TOTAL); colormap('jet');  subplot(2,1,2); plot(label_TOTAL);   
        
        %save(sprintf('/user/chae2089/CWpaper/features_4/subject%d_PSD_delta.mat',n_p), 'TOTAL_delta');
        %save(sprintf('/user/chae2089/CWpaper/features_4/subject%d_PSD_theta.mat',n_p), 'TOTAL_theta');
        %save(sprintf('/user/chae2089/CWpaper/features_4/subject%d_PSD_alpha.mat',n_p), 'TOTAL_alpha');
        %save(sprintf('/user/chae2089/CWpaper/features_4/subject%d_PSD_beta.mat',n_p), 'TOTAL_beta');
        %save(sprintf('/user/chae2089/CWpaper/features_4/subject%d_PSD_gamma.mat',n_p), 'TOTAL_gamma');
        save(sprintf('/user/chae2089/CWpaper/FN_REVISION/feature_for_sleep_stage/subject%d_PSD_TOTAL.mat',n_p), 'TOTAL');
        save(sprintf('/user/chae2089/CWpaper/FN_REVISION/feature_for_sleep_stage/subject%d_PSD_TOTAL_label.mat',n_p), 'label_TOTAL');
        
        
	else               
		
        % subjects with all stages
        stage_W = reshape(stage_W,6,6000*size(stage_W,3));
        stage_N1 = reshape(stage_N1,6,6000*size(stage_N1,3));
        stage_N2 = reshape(stage_N2,6,6000*size(stage_N2,3));
        stage_N3 = reshape(stage_N3,6,6000*size(stage_N3,3));
        stage_R = reshape(stage_R,6,6000*size(stage_R,3));
        save(sprintf('/user/chae2089/CWpaper/stage_data/subject%d_W.mat',n_p), 'stage_W');
        save(sprintf('/user/chae2089/CWpaper/stage_data/subject%d_N1.mat',n_p), 'stage_N1');
        save(sprintf('/user/chae2089/CWpaper/stage_data/subject%d_N2.mat',n_p), 'stage_N2');
        save(sprintf('/user/chae2089/CWpaper/stage_data/subject%d_N3.mat',n_p), 'stage_N3');
        save(sprintf('/user/chae2089/CWpaper/stage_data/subject%d_R.mat',n_p), 'stage_R');
        
        % preprocessing - filtering 0.5-50H via EEGLAB
        clear EEG;
        EEG.etc.eeglabvers = '2022.0'; % this tracks which version of EEGLAB is being used, you may ignore it
        EEG = pop_importdata('dataformat','matlab','nbchan',0,'data',sprintf('/user/chae2089/CWpaper/stage_data/subject%d_W.mat',n_p),'srate',200,'pnts',0,'xmin',0);
        EEG.setname='W';
        EEG = eeg_checkset( EEG );
        EEG = pop_eegfiltnew(EEG, 'locutoff',0.5);
        EEG = eeg_checkset( EEG );
        EEG = pop_eegfiltnew(EEG, 'hicutoff',50);
        EEG = eeg_checkset( EEG );
        W=EEG.data;
        
        clear EEG;
        EEG.etc.eeglabvers = '2022.0'; % this tracks which version of EEGLAB is being used, you may ignore it
        EEG = pop_importdata('dataformat','matlab','nbchan',0,'data',sprintf('/user/chae2089/CWpaper/stage_data/subject%d_R.mat',n_p),'srate',200,'pnts',0,'xmin',0);
        EEG.setname='R';
        EEG = eeg_checkset( EEG );
        EEG = pop_eegfiltnew(EEG, 'locutoff',0.5);
        EEG = eeg_checkset( EEG );
        EEG = pop_eegfiltnew(EEG, 'hicutoff',50);
        EEG = eeg_checkset( EEG );
        R=EEG.data;
        
        clear EEG;
        EEG.etc.eeglabvers = '2022.0'; % this tracks which version of EEGLAB is being used, you may ignore it
        EEG = pop_importdata('dataformat','matlab','nbchan',0,'data',sprintf('/user/chae2089/CWpaper/stage_data/subject%d_N1.mat',n_p),'srate',200,'pnts',0,'xmin',0);
        EEG.setname='N1';
        EEG = eeg_checkset( EEG );
        EEG = pop_eegfiltnew(EEG, 'locutoff',0.5);
        EEG = eeg_checkset( EEG );
        EEG = pop_eegfiltnew(EEG, 'hicutoff',50);
        EEG = eeg_checkset( EEG );
        N1=EEG.data;

        clear EEG;
        EEG.etc.eeglabvers = '2022.0'; % this tracks which version of EEGLAB is being used, you may ignore it
        EEG = pop_importdata('dataformat','matlab','nbchan',0,'data',sprintf('/user/chae2089/CWpaper/stage_data/subject%d_N2.mat',n_p),'srate',200,'pnts',0,'xmin',0);
        EEG.setname='N2';
        EEG = eeg_checkset( EEG );
        EEG = pop_eegfiltnew(EEG, 'locutoff',0.5);
        EEG = eeg_checkset( EEG );
        EEG = pop_eegfiltnew(EEG, 'hicutoff',50);
        EEG = eeg_checkset( EEG );
        N2=EEG.data;
        
        clear EEG;
        EEG.etc.eeglabvers = '2022.0'; % this tracks which version of EEGLAB is being used, you may ignore it
        EEG = pop_importdata('dataformat','matlab','nbchan',0,'data',sprintf('/user/chae2089/CWpaper/stage_data/subject%d_N3.mat',n_p),'srate',200,'pnts',0,'xmin',0);
        EEG.setname='N3';
        EEG = eeg_checkset( EEG );
        EEG = pop_eegfiltnew(EEG, 'locutoff',0.5);
        EEG = eeg_checkset( EEG );
        EEG = pop_eegfiltnew(EEG, 'hicutoff',50);
        EEG = eeg_checkset( EEG );
        N3=EEG.data;
        
        
        % PSD feature extraction of 6 channels
        W_delta_l2=[]; W_delta_K1_2=[]; W_delta_K2_2=[]; W_theta2=[]; W_alpha2=[]; W_spin1_2=[]; W_spin2_2=[]; W_beta2=[]; 
        R_delta_l2=[]; R_delta_K1_2=[]; R_delta_K2_2=[]; R_theta2=[]; R_alpha2=[]; R_spin1_2=[]; R_spin2_2=[]; R_beta2=[]; 
        N1_delta_l2=[]; N1_delta_K1_2=[]; N1_delta_K2_2=[]; N1_theta2=[]; N1_alpha2=[]; N1_spin1_2=[]; N1_spin2_2=[]; N1_beta2=[]; 
        N2_delta_l2=[]; N2_delta_K1_2=[]; N2_delta_K2_2=[]; N2_theta2=[]; N2_alpha2=[]; N2_spin1_2=[]; N2_spin2_2=[]; N2_beta2=[]; 
        N3_delta_l2=[]; N3_delta_K1_2=[]; N3_delta_K2_2=[]; N3_theta2=[]; N3_alpha2=[]; N3_spin1_2=[]; N3_spin2_2=[]; N3_beta2=[]; 
    
        for ch=1:6
			%if ch==1 || ch==2 || ch==5 || ch==6
            [S_w,F_w,T_w,P_w] = spectrogram(W(ch,:), kaiser(200*1,0.5),100,2^10,200,'yaxis'); P_w=P_w./sum(P_w(4:155,:),1); %(4:155,:)
            W_delta_l=[]; W_delta_K1=[]; W_delta_K2=[]; W_theta=[]; W_alpha=[]; W_spin1=[]; W_spin2=[]; W_beta=[]; 

            for t1=1:60:size(T_w,2)-31
                W_K_comp=sort(sum(P_w(7:21,t1:t1+58))); W_spindle=sort(sum(P_w(63:77,t1:t1+58)));
                W_delta_l=[W_delta_l,mean(sum(P_w(4:11,t1:t1+58)))]; W_delta_K1=[W_delta_K1,W_K_comp(59)]; W_delta_K2=[W_delta_K2,mean(W_K_comp(1:58))]; 
                W_theta=[W_theta,mean(sum(P_w(22:41,t1:t1+58)))]; W_alpha=[W_alpha,mean(sum(P_w(42:62,t1:t1+58)))]; 
                W_spin1=[W_spin1,W_spindle(59)]; W_spin2=[W_spin2,mean(W_spindle(1:58))]; W_beta=[W_beta,mean(sum(P_w(78:155,t1:t1+58)))]; 
            end

            [S_r,F_r,T_r,P_r] = spectrogram(R(ch,:), kaiser(200*1,0.5),100,2^10,200,'yaxis'); P_r=P_r./sum(P_r(4:155,:),1);
            R_delta_l=[]; R_delta_K1=[]; R_delta_K2=[]; R_theta=[]; R_alpha=[]; R_spin1=[]; R_spin2=[]; R_beta=[]; 

            for t1=1:60:size(T_r,2)-31
                R_K_comp=sort(sum(P_r(7:21,t1:t1+58))); R_spindle=sort(sum(P_r(63:77,t1:t1+58)));
                R_delta_l=[R_delta_l,mean(sum(P_r(4:11,t1:t1+58)))]; R_delta_K1=[R_delta_K1,R_K_comp(59)]; R_delta_K2=[R_delta_K2,mean(R_K_comp(1:58))]; 
                R_theta=[R_theta,mean(sum(P_r(22:41,t1:t1+58)))]; R_alpha=[R_alpha,mean(sum(P_r(42:62,t1:t1+58)))]; 
                R_spin1=[R_spin1,R_spindle(59)]; R_spin2=[R_spin2,mean(R_spindle(1:58))]; R_beta=[R_beta,mean(sum(P_r(78:155,t1:t1+58)))]; 
            end       

            [S_n1,F_n1,T_n1,P_n1] = spectrogram(N1(ch,:), kaiser(200*1,0.5),100,2^10,200,'yaxis'); P_n1=P_n1./sum(P_n1(4:155,:),1);
            N1_delta_l=[]; N1_delta_K1=[]; N1_delta_K2=[]; N1_theta=[]; N1_alpha=[]; N1_spin1=[]; N1_spin2=[]; N1_beta=[]; 

            for t1=1:60:size(T_n1,2)-31
                N1_K_comp=sort(sum(P_n1(7:21,t1:t1+58))); N1_spindle=sort(sum(P_n1(63:77,t1:t1+58)));
                N1_delta_l=[N1_delta_l,mean(sum(P_n1(4:11,t1:t1+58)))]; N1_delta_K1=[N1_delta_K1,N1_K_comp(59)]; N1_delta_K2=[N1_delta_K2,mean(N1_K_comp(1:58))]; 
                N1_theta=[N1_theta,mean(sum(P_n1(22:41,t1:t1+58)))]; N1_alpha=[N1_alpha,mean(sum(P_n1(42:62,t1:t1+58)))]; 
                N1_spin1=[N1_spin1,N1_spindle(59)]; N1_spin2=[N1_spin2,mean(N1_spindle(1:58))]; N1_beta=[N1_beta,mean(sum(P_n1(78:155,t1:t1+58)))]; 
            end

            [S_n2,F_n2,T_n2,P_n2] = spectrogram(N2(ch,:), kaiser(200*1,0.5),100,2^10,200,'yaxis'); P_n2=P_n2./sum(P_n2(4:155,:),1);
            N2_delta_l=[]; N2_delta_K1=[]; N2_delta_K2=[]; N2_theta=[]; N2_alpha=[]; N2_spin1=[]; N2_spin2=[]; N2_beta=[]; 

            for t1=1:60:size(T_n2,2)-31
                N2_K_comp=sort(sum(P_n2(7:21,t1:t1+58))); N2_spindle=sort(sum(P_n2(63:77,t1:t1+58)));
                N2_delta_l=[N2_delta_l,mean(sum(P_n2(4:11,t1:t1+58)))]; N2_delta_K1=[N2_delta_K1,N2_K_comp(59)]; N2_delta_K2=[N2_delta_K2,mean(N2_K_comp(1:58))]; 
                N2_theta=[N2_theta,mean(sum(P_n2(22:41,t1:t1+58)))]; N2_alpha=[N2_alpha,mean(sum(P_n2(42:62,t1:t1+58)))]; 
                N2_spin1=[N2_spin1,N2_spindle(59)]; N2_spin2=[N2_spin2,mean(N2_spindle(1:58))]; N2_beta=[N2_beta,mean(sum(P_n2(78:155,t1:t1+58)))]; 
            end
            
            [S_n3,F_n3,T_n3,P_n3] = spectrogram(N3(ch,:), kaiser(200*1,0.5),100,2^10,200,'yaxis'); P_n3=P_n3./sum(P_n3(4:155,:),1);
            N3_delta_l=[]; N3_delta_K1=[]; N3_delta_K2=[]; N3_theta=[]; N3_alpha=[]; N3_spin1=[]; N3_spin2=[]; N3_beta=[]; 

            for t1=1:60:size(T_n3,2)-31
                N3_K_comp=sort(sum(P_n3(7:21,t1:t1+58))); N3_spindle=sort(sum(P_n3(63:77,t1:t1+58)));
                N3_delta_l=[N3_delta_l,mean(sum(P_n3(4:11,t1:t1+58)))]; N3_delta_K1=[N3_delta_K1,N3_K_comp(59)]; N3_delta_K2=[N3_delta_K2,mean(N3_K_comp(1:58))]; 
                N3_theta=[N3_theta,mean(sum(P_n3(22:41,t1:t1+58)))]; N3_alpha=[N3_alpha,mean(sum(P_n3(42:62,t1:t1+58)))]; 
                N3_spin1=[N3_spin1,N3_spindle(59)]; N3_spin2=[N3_spin2,mean(N3_spindle(1:58))]; N3_beta=[N3_beta,mean(sum(P_n3(78:155,t1:t1+58)))]; 
            end
            
            % cumulate features
            W_delta_l2=[W_delta_l2; W_delta_l]; W_delta_K1_2=[W_delta_K1_2; W_delta_K1]; W_delta_K2_2=[W_delta_K2_2; W_delta_K2]; W_theta2=[W_theta2; W_theta]; 
            W_alpha2=[W_alpha2; W_alpha]; W_spin1_2=[W_spin1_2; W_spin1]; W_spin2_2=[W_spin2_2; W_spin2]; W_beta2=[W_beta2; W_beta]; 
            
            R_delta_l2=[R_delta_l2; R_delta_l]; R_delta_K1_2=[R_delta_K1_2; R_delta_K1]; R_delta_K2_2=[R_delta_K2_2; R_delta_K2]; R_theta2=[R_theta2; R_theta]; 
            R_alpha2=[R_alpha2; R_alpha]; R_spin1_2=[R_spin1_2; R_spin1]; R_spin2_2=[R_spin2_2; R_spin2]; R_beta2=[R_beta2; R_beta]; 
            
            N1_delta_l2=[N1_delta_l2; N1_delta_l]; N1_delta_K1_2=[N1_delta_K1_2; N1_delta_K1]; N1_delta_K2_2=[N1_delta_K2_2; N1_delta_K2]; N1_theta2=[N1_theta2; N1_theta]; 
            N1_alpha2=[N1_alpha2; N1_alpha]; N1_spin1_2=[N1_spin1_2; N1_spin1]; N1_spin2_2=[N1_spin2_2; N1_spin2]; N1_beta2=[N1_beta2; N1_beta]; 
            
            N2_delta_l2=[N2_delta_l2; N2_delta_l]; N2_delta_K1_2=[N2_delta_K1_2; N2_delta_K1]; N2_delta_K2_2=[N2_delta_K2_2; N2_delta_K2]; N2_theta2=[N2_theta2; N2_theta]; 
            N2_alpha2=[N2_alpha2; N2_alpha]; N2_spin1_2=[N2_spin1_2; N2_spin1]; N2_spin2_2=[N2_spin2_2; N2_spin2]; N2_beta2=[N2_beta2; N2_beta]; 
            
            N3_delta_l2=[N3_delta_l2; N3_delta_l]; N3_delta_K1_2=[N3_delta_K1_2; N3_delta_K1]; N3_delta_K2_2=[N3_delta_K2_2; N3_delta_K2]; N3_theta2=[N3_theta2; N3_theta]; 
            N3_alpha2=[N3_alpha2; N3_alpha]; N3_spin1_2=[N3_spin1_2; N3_spin1]; N3_spin2_2=[N3_spin2_2; N3_spin2]; N3_beta2=[N3_beta2; N3_beta]; 
        end
        
        % save features and their labels
        TOTAL_delta_l = [W_delta_l2, N1_delta_l2, N2_delta_l2, N3_delta_l2, R_delta_l2];
        TOTAL_delta_K1 = [W_delta_K1_2, N1_delta_K1_2, N2_delta_K1_2, N3_delta_K1_2, R_delta_K1_2];
        TOTAL_delta_K2 = [W_delta_K2_2, N1_delta_K2_2, N2_delta_K2_2, N3_delta_K2_2, R_delta_K2_2];
        TOTAL_theta = [W_theta2, N1_theta2, N2_theta2, N3_theta2, R_theta2];
        TOTAL_alpha = [W_alpha2, N1_alpha2, N2_alpha2, N3_alpha2, R_alpha2];
        TOTAL_spin1 = [W_spin1_2, N1_spin1_2, N2_spin1_2, N3_spin1_2, R_spin1_2];
        TOTAL_spin2 = [W_spin2_2, N1_spin2_2, N2_spin2_2, N3_spin2_2, R_spin2_2];
        TOTAL_beta = [W_beta2, N1_beta2, N2_beta2, N3_beta2, R_beta2];
        %TOTAL_gamma = [W_spindle2, N1_spindle2, N2_spindle2, R_spindle2];
        
        % save features and their labels
        TOTAL=[TOTAL_delta_l; TOTAL_delta_K1; TOTAL_delta_K2; TOTAL_theta; TOTAL_alpha; TOTAL_spin1; TOTAL_spin2; TOTAL_beta];
        label_TOTAL =[ones(1,size(W_delta_l2,2)), 2*ones(1,size(N1_delta_l2,2)), 3*ones(1,size(N2_delta_l2,2)), 4*ones(1,size(N3_delta_l2,2)), 5*ones(1,size(R_delta_l2,2))]; 
        
        %figure(12); subplot(2,1,1); imagesc(TOTAL); colormap('jet');  subplot(2,1,2); plot(label_TOTAL); 
                
        %save(sprintf('/user/chae2089/CWpaper/features_4/subject%d_PSD_delta.mat',n_p), 'TOTAL_delta');
        %save(sprintf('/user/chae2089/CWpaper/features_4/subject%d_PSD_theta.mat',n_p), 'TOTAL_theta');
        %save(sprintf('/user/chae2089/CWpaper/features_4/subject%d_PSD_alpha.mat',n_p), 'TOTAL_alpha');
        %save(sprintf('/user/chae2089/CWpaper/features_4/subject%d_PSD_beta.mat',n_p), 'TOTAL_beta');
        %save(sprintf('/user/chae2089/CWpaper/features_4/subject%d_PSD_gamma.mat',n_p), 'TOTAL_gamma');
        save(sprintf('/user/chae2089/CWpaper/FN_REVISION/feature_for_sleep_stage/subject%d_PSD_TOTAL.mat',n_p), 'TOTAL');
        save(sprintf('/user/chae2089/CWpaper/FN_REVISION/feature_for_sleep_stage/subject%d_PSD_TOTAL_label.mat',n_p), 'label_TOTAL');
        
          
            
    end
end

    
    
    