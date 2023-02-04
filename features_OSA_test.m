%% OSA feature extraction script
% Sora An (soraan@ewha.ac.kr); Chaewon Kang (chaewonkang46@gmail.com); 2022


clear all;

T=readtable('/user/chae2089/CWpaper/SDP_osa.xlsx','Sheet','Sheet1'); %%% set the file path of "SDP.xlsx"
fs=200;

normal_Y_test = [29, 90, 119, 140];
normal_O_test = [164,165,167,168];
mtom_Y_test = [54, 80,88,107];
mtom_O_test= [16,72,93,103,121,145];
severe_Y_test = [132,175,176,177,179];
severe_O_test = [52,169,170,171,174];
all =[normal_Y_test,normal_O_test,mtom_Y_test,mtom_O_test,severe_Y_test, severe_O_test]; 



for n1=1:length(all)
	data1=load(sprintf('/user/chae2089/CWpaper/FN_REVISION/feature_for_sleep_stage/subject%d_PSD_TOTAL.mat',all(n1))); data1=data1.TOTAL;
	data1_label=readtable(sprintf('/user/chae2089/CWpaper/FN_REVISION/GENERAL_MODEL_num_stage_fix/MLP/OUT/pred_%d.csv',all(n1))); data1_label=(data1_label.Var1)';
	
	%data1_label=load(sprintf('/user/chae2089/CWpaper/FN_REVISION/feature_for_sleep_stage/subject%d_PSD_TOTAL_label.mat',all(n1))); data1_label=data1_label.label_TOTAL;
	%data1_label=readtable(sprintf('/user/chae2089/CWpaper/FINAL_GENERAL_MODEL_stage_num_fix/pred_stage/pred_%d.csv',all(n1))); data1_label=(data1_label.Var1)';
	%figure(1); plot(data1_label_orig,'r'); hold on; plot(data1_label,'b'); hold off;
	
	nrem_ind=find(data1_label==2 | data1_label==3 | data1_label==4);
	rem_ind=find(data1_label==5);
	    
	nrem_rem=[mean(data1(1:48,nrem_ind),2); mean(data1(1:48,rem_ind),2)];
	nrem_rem_comb_spo2=[mean(data1(1:48,nrem_ind),2); mean(data1(1:48,rem_ind),2); T.Var11(all(n1))];
	%beta_nrem_rem=[st_r'; T.Var10(all(n1)); T.Var11(all(n1))];
	%c_index=[T.Var10(all(n1));T.Var11(all(n1))];	

	nrem_rem_all=[mean(data1(1:48,nrem_ind),2); mean(data1(1:48,rem_ind),2); st_r'];
	nrem_rem_all_comb_spo2=[mean(data1(1:48,nrem_ind),2); mean(data1(1:48,rem_ind),2); st_r'; T.Var11(all(n1))];
	
	save(sprintf('/user/chae2089/CWpaper/FN_REVISION/feature_for_2nd_model/results_from_1st_model/MLP/subject%d.mat',all(n1)), 'nrem_rem');
	save(sprintf('/user/chae2089/CWpaper/FN_REVISION/feature_for_2nd_model/eeg_combined_spo2/subject%d.mat',all(n1)), 'nrem_rem_all_comb_spo2');

	
	%st_r_int=[st_r_int, st_r'];
	%data11=[data11, beta_nrem_rem ];	
	
	all(n1)
	
end

		