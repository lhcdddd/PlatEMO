function Next = RSurrogateAssistedSelection(Problem,Ref,Input,wmax,Smodel,ArchiveDecs)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Next = OperatorGA(Problem,[Input;Ref.decs],{1,15,1,5});
    i    = 0;
    while i < wmax
        [soerted_index,~] = model_select(Smodel,Next,ArchiveDecs);
        Input = Next(soerted_index(1:length(Ref)),:);
        Next  = OperatorGA(Problem,[Input;Ref.decs],{1,15,1,5});
        i     = i + size(Next,1);
    end
    [~,scores] = model_select(Smodel,Next,ArchiveDecs);
    [~,ind] = sort(scores,'descend');
    Next    = Next(ind(1:min(4, length(ind))),:); 
end

function [ind,Final_Score] = model_select(Smodel,Next,ArchiveDecs)
    ksc = Smodel.ksc;
    mode = Smodel.mode;
    Next_num = size(Next,1);
    
    if ksc > 0
        model_x = Smodel.X;    
        C1_data = model_x(Smodel.Y ==1,:);
        C2_data = model_x(Smodel.Y ~=1,:);

        C1_num   = size(C1_data,1);
        C2_num   = size(C2_data,1);

        scores = zeros(Next_num,2);
        
        all_testdata = zeros(2*(C1_num+C2_num)*Next_num,2*size(C1_data,2));
        for i = 1 : size(Next,1)
            original = (i-1)*2*(C1_num+C2_num);
            Xi       = repmat(Next(i,:),size(C1_data,1),1);
            all_testdata(original+1:original+C1_num,:)          = [C1_data,Xi];  %C1_Xi
            all_testdata(original+1+C1_num:original+C1_num*2,:) = [Xi,C1_data]; %Xi_C1
            
            Xi = repmat(Next(i,:),size(C2_data,1),1);
            all_testdata(original+1+C1_num*2:original+C1_num*2+C2_num,:)          = [C2_data,Xi]; %C2_Xi
            all_testdata(original+1+C2_num+C1_num*2:original+C2_num*2+C1_num*2,:) = [Xi,C2_data];%Xi_C2
        end
        
        TestIn_nor = mapminmax('apply',all_testdata',Smodel.mp_struct)';
        pre_out    = Smodel.net(TestIn_nor')';  
        
        for i = 1 : size(Next,1)
            C_SCORE    = zeros(1,2);
            original   = (i-1)*2*(C1_num+C2_num);
            pre_C1Xi   = sum(pre_out(original+1:original+C1_num,:),1)./C1_num;
            C_SCORE(1) = C_SCORE(1) + pre_C1Xi(2)+pre_C1Xi(3);   
            C_SCORE(2) = C_SCORE(2) + pre_C1Xi(1);               
            
            pre_XiC1   = sum(pre_out(original+1+C1_num:original+C1_num*2,:),1)./C1_num;
            C_SCORE(1) = C_SCORE(1) + pre_XiC1(2) + pre_XiC1(1);  
            C_SCORE(2) = C_SCORE(2) + pre_XiC1(3);                 
            
            pre_C2Xi   = sum(pre_out(original+1+C1_num*2:original+C1_num*2+C2_num,:),1)./C2_num;
            C_SCORE(1) = C_SCORE(1) + pre_C2Xi(3);
            C_SCORE(2) = C_SCORE(2) + pre_C2Xi(2) + pre_C2Xi(1);
            
            pre_XiC2   = sum(pre_out(original+1+C2_num+C1_num*2:original+C2_num*2+C1_num*2,:),1)./C2_num;
            C_SCORE(1) = C_SCORE(1) + pre_XiC2(1);
            C_SCORE(2) = C_SCORE(2) + pre_XiC2(2) + pre_XiC2(3);
            
            scores(i,1) = C_SCORE(1)-C_SCORE(2);
        end
        P_conv = scores(:,1);
        if max(P_conv) > min(P_conv)
            P_conv = (P_conv - min(P_conv)) / (max(P_conv) - min(P_conv));
        else
            P_conv = ones(size(P_conv));
        end
        
        % R2: Reverse mode handling (Invert the convergence score)
        if mode == -1
            P_conv = 1 - P_conv;
        end
    else
        P_conv = zeros(Next_num, 1);
    end
    
    if ksc < 1
        Distance = pdist2(Next, ArchiveDecs);
        P_div = min(Distance, [], 2);
        if max(P_div) > min(P_div)
            P_div = (P_div - min(P_div)) / (max(P_div) - min(P_div));
        else
            P_div = ones(size(P_div));
        end
    else
        P_div = zeros(Next_num, 1);
    end
    
    Final_Score = ksc * P_conv + (1 - ksc) * P_div;
    [~,ind] = sort(Final_Score,'descend');  
    
    % Append info to a log file
    fileID = fopen('e:\PlatEMO\REMOss_debug_log.txt', 'a');
    fprintf(fileID, '  Selected 4 solutions info (Mode %d):\n', mode);
    for idx = 1:min(4, length(ind))
        sel_idx = ind(idx);
        fprintf(fileID, '    Sol %d: Final_Score=%.4f (P_conv=%.4f, P_div=%.4f)\n', ...
            idx, Final_Score(sel_idx), P_conv(sel_idx), P_div(sel_idx));
    end
    fclose(fileID);
end