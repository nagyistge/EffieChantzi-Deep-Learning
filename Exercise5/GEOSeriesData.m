        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Chantzi Efthymia - Deep Learning - Exercises 5 %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function retrieves a GEO series record dataset from GEO database %
% in GEO series text format file, parses it and extracts the 2-D matrix %
% of expression data, where rows respond to samples and columns to the  %
% different genes.                                                      %
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% GSE_id: unique GEO accession number for a series record gene          %
% expression dataset                                                    %
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% expressionMatrix: 2-D gene expression (num_of_samplesxnum_of_genes)   %
% matrix                                                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function expressionMatrix = GEOSeriesData(GSE_id)

suffix = '.txt';
gse_file = strcat(GSE_id, suffix);

gseData = getgeodata(GSE_id, 'ToFile', gse_file);
gseData = geoseriesread(gse_file);

data = gseData.Data;

genes = data.NRows;
samples = data.NCols;
col = data.ColNames;

expressionMatrix = zeros(samples, genes);
for i = 1 : samples
    
    expressionMatrix(i, :) = data.(':')(col{i});
    
end

end
