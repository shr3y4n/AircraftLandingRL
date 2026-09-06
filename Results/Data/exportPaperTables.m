function exportPaperTables(summary_data, out_dir)
% EXPORTPAPERTABLES Generates LaTeX tables and CSV data suitable for research publications.
%
% Formats quantitative landing dispersion metrics, tracking errors, and success rates
% into publication-ready LaTeX tabular code.
%
% Inputs:
%   summary_data - Monte Carlo summary struct from runMonteCarlo()
%   out_dir      - Output folder (defaults to Results/Data/)

if nargin < 2 || isempty(out_dir)
    out_dir = fullfile(pwd, 'Results', 'Data');
end
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

if nargin < 1 || isempty(summary_data)
    mat_file = fullfile(out_dir, 'monte_carlo_results.mat');
    if exist(mat_file, 'file')
        loaded = load(mat_file);
        summary_data = loaded.summary;
    else
        error('exportPaperTables:NoData', 'No summary data provided and %s not found.', mat_file);
    end
end

tex_file = fullfile(out_dir, 'paper_landing_metrics_table.tex');
fid = fopen(tex_file, 'w');
if fid < 0
    error('Could not open %s for writing.', tex_file);
end

fprintf(fid, '%% Auto-generated Table of Quantitative Landing Performance\n');
fprintf(fid, '\\begin{table*}[t]\n');
fprintf(fid, '\\centering\n');
fprintf(fid, '\\caption{Quantitative Flight Performance & Touchdown Robustness Comparison across Experimental Suites}\n');
fprintf(fid, '\\label{tab:landing_performance}\n');
fprintf(fid, '\\begin{tabular}{l c c c c c c}\n');
fprintf(fid, '\\hline\\hline\n');
fprintf(fid, 'Experimental Suite & Success [\\%%] & $\\Delta x_{\\text{td}}$ [m] & $\\dot{h}_{\\text{td}}$ [m/s] & $\\text{RMS}(e_h)$ [m] & $\\text{RMS}(e_V)$ [m/s] & Control Effort \\\\\n');
fprintf(fid, '\\hline\n');

for s = 1:length(summary_data)
    d = summary_data(s);
    fprintf(fid, '%-26s & %5.1f & $%+6.2f \\pm %5.2f$ & $%5.2f \\pm %4.2f$ & $%5.2f \\pm %4.2f$ & $%5.2f \\pm %4.2f$ & $%6.1f \\pm %5.1f$ \\\\\n', ...
            d.name, d.success_rate, d.mean_pos_err, d.std_pos_err, ...
            d.mean_sink_rate, d.std_sink_rate, ...
            d.mean_rms_eh, d.std_rms_eh, ...
            d.mean_rms_eV, d.std_rms_eV, ...
            d.mean_effort, d.std_effort);
end

fprintf(fid, '\\hline\\hline\n');
fprintf(fid, '\\end{tabular}\n');
fprintf(fid, '\\end{table*}\n');
fclose(fid);

fprintf('[Paper Export] LaTeX table exported to: %s\n', tex_file);

end
