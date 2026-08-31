function results = run_hdrvdp3_dir(gen_dir, ref_dir, varargin)
%RUN_HDRVDP3_DIR  Evaluate HDR-VDP-3 between a generated folder and a reference folder.
%
% Usage (MATLAB):
%   results = run_hdrvdp3_dir(gen_dir, ref_dir, ...
%       'GenPattern',  {'*.exr','*.hdr'}, ...
%       'RefPattern',  '*.exr', ...
%       'PPD',         60, ...
%       'ColorEncoding','luminance', ...
%       'LPeak',       1000, ...
%       'ScaleMode',   'max', ...      % 'max' or 'percentile'
%       'ScaleP',      99.9, ...
%       'OutCSV',      '', ...
%       'SaveMapsDir', '', ...
%       'Quiet',       false);
%
% Notes:
% - gen files may be .exr or .hdr; reference is usually .exr.
% - Pairing is by stem (file name without extension), e.g., "001".
% - For gen .hdr, reference is still ref_dir/"001.exr" (same stem).
%
% Recommended CLI:
%   matlab -batch "addpath(genpath('/path/to/HDRVDP3_Eval')); run_hdrvdp3_dir('...gen...','...ref...','OutCSV','out.csv','Quiet',true);"

p = inputParser;
p.addRequired('gen_dir', @(s)ischar(s)||isstring(s));
p.addRequired('ref_dir', @(s)ischar(s)||isstring(s));
p.addParameter('GenPattern', {'*.exr','*.hdr'}, @(x)iscell(x)||ischar(x)||isstring(x));
p.addParameter('RefPattern', '*.exr', @(s)ischar(s)||isstring(s));
p.addParameter('PPD', 77, @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('ColorEncoding', 'luminance', @(s)ischar(s)||isstring(s));
p.addParameter('Options', {}, @(c)iscell(c));
p.addParameter('LPeak', 1000, @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('ScaleMode', 'max', @(s)ischar(s)||isstring(s));  % 'max'|'percentile'
p.addParameter('ScaleP', 99.9, @(x)isnumeric(x)&&isscalar(x)&&x>0&&x<=100);
p.addParameter('OutCSV', '', @(s)ischar(s)||isstring(s));
p.addParameter('SaveMapsDir', '', @(s)ischar(s)||isstring(s));
p.addParameter('Quiet', false, @(b)islogical(b)||ismember(b,[0 1]));
p.addParameter('Ours', false, @(b)islogical(b)||ismember(b,[0 1]));
p.addParameter('OursFrameName', 'frame_00.exr', @(s)ischar(s)||isstring(s));
p.parse(gen_dir, ref_dir, varargin{:});

gen_dir        = char(p.Results.gen_dir);
ref_dir        = char(p.Results.ref_dir);
gen_pattern    = p.Results.GenPattern;
ref_pattern    = char(p.Results.RefPattern);
ppd            = p.Results.PPD;
color_encoding = char(p.Results.ColorEncoding);
opt_cell       = p.Results.Options;
L_peak         = p.Results.LPeak;
scale_mode     = lower(char(p.Results.ScaleMode));
scale_p        = p.Results.ScaleP;
out_csv        = char(p.Results.OutCSV);
maps_dir       = char(p.Results.SaveMapsDir);
quiet          = logical(p.Results.Quiet);
ours           = logical(p.Results.Ours);
ours_frame     = char(p.Results.OursFrameName);

% ------------------ list files ------------------
%if ischar(gen_pattern) || isstring(gen_pattern)
%    gen_pattern = {char(gen_pattern)};
%end

%gen_list = [];
%for i = 1:numel(gen_pattern)
%    gen_list = [gen_list; dir(fullfile(gen_dir, char(gen_pattern{i})))]; %#ok<AGROW>
%end
%ref_list = dir(fullfile(ref_dir, ref_pattern));

% ------------------ list files ------------------
if ~ours
    if ischar(gen_pattern) || isstring(gen_pattern)
        gen_pattern = {char(gen_pattern)};
    end

    gen_list = [];
    for i = 1:numel(gen_pattern)
        gen_list = [gen_list; dir(fullfile(gen_dir, char(gen_pattern{i})))]; %#ok<AGROW>
    end
else
    % Ours mode: recursively find .../<stem>/<frame_00.exr>
    % (requires MATLAB supporting **, e.g. R2016b+)
    gen_list = dir(fullfile(gen_dir, '**', ours_frame));
end

ref_list = dir(fullfile(ref_dir, ref_pattern));

if isempty(gen_list)
    error('No gen files found under "%s" with pattern(s).', gen_dir);
end
if isempty(ref_list)
    error('No ref files found under "%s" with pattern "%s".', ref_dir, ref_pattern);
end

% build stem->path maps
%gen_map = containers.Map;
%for k = 1:numel(gen_list)
%    [~, stem] = fileparts(gen_list(k).name);
%    if startsWith(stem,'._'), continue; end
%    gen_map(stem) = fullfile(gen_dir, gen_list(k).name);
%end

gen_map = containers.Map;
for k = 1:numel(gen_list)
    if ~ours
        [~, stem] = fileparts(gen_list(k).name);
        if startsWith(stem,'._'), continue; end
        gen_map(stem) = fullfile(gen_dir, gen_list(k).name);
    else
        % gen_list(k) in ours mode already contains folder info
        % Use the parent folder name as stem: .../<stem>/frame_00.exr
        if startsWith(gen_list(k).name, '._'), continue; end
        gp_full = fullfile(gen_list(k).folder, gen_list(k).name);
        [parent_dir, ~] = fileparts(gp_full);   % remove filename -> .../<stem>
        [~, stem] = fileparts(parent_dir);      % stem = '001'
        if startsWith(stem,'._'), continue; end
        gen_map(stem) = gp_full;
    end
end

ref_map = containers.Map;
for k = 1:numel(ref_list)
    [~, stem] = fileparts(ref_list(k).name);
    if startsWith(stem,'._'), continue; end
    ref_map(stem) = fullfile(ref_dir, ref_list(k).name);
end

common = sort(intersect(gen_map.keys, ref_map.keys));
if isempty(common)
    % helpful hint: show a few stems
    gk = gen_map.keys; rk = ref_map.keys;
    gk = gk(1:min(10,numel(gk))); rk = rk(1:min(10,numel(rk)));
    error("No paired files by stem.\nExample gen stems: %s\nExample ref stems: %s", ...
        strjoin(string(gk), ", "), strjoin(string(rk), ", "));
end

if ~isempty(maps_dir) && ~exist(maps_dir, 'dir')
    mkdir(maps_dir);
end

% outputs
names  = strings(numel(common),1);
gpaths = strings(numel(common),1);
rpaths = strings(numel(common),1);
P_det  = nan(numel(common),1);
C_max  = nan(numel(common),1);
Q      = nan(numel(common),1);
Q_JOD  = nan(numel(common),1);

% ------------------ loop ------------------
for i = 1:numel(common)
    stem = common{i};
    gp = gen_map(stem);
    rp = ref_map(stem);

    if ~quiet
        fprintf('[%4d/%4d] %s\n  gen=%s\n  ref=%s\n', i, numel(common), stem, gp, rp);
    end

    gen_img = read_hdr_any(gp);   % HxWxC single
    ref_img = exrread(rp);        % gt is exr

    % convert to luminance (HDR-VDP supports 'luminance' efficiently)
    gen_L = rgb2luminance_709(gen_img);
    ref_L = rgb2luminance_709(ref_img);

    % map relative -> absolute cd/m^2
    switch scale_mode
        case 'max'
            gen_L = gen_L / max(gen_L(:)) * L_peak;
            ref_L = ref_L / max(ref_L(:)) * L_peak;
        case 'percentile'
            gen_L = scale_to_peak_percentile(gen_L, L_peak, scale_p, 1e-6);
            ref_L = scale_to_peak_percentile(ref_L, L_peak, scale_p, 1e-6);
        otherwise
            error('Unknown ScaleMode="%s". Use "max" or "percentile".', scale_mode);
    end

    % size match
    if ~isequal(size(gen_L), size(ref_L))
        % If your dataset is consistent, you might prefer error instead of crop.
        % Here we center-crop the larger one to the smaller one's size.
        [gen_L, ref_L] = crop_to_match(gen_L, ref_L);
        if ~quiet
            warning('Size mismatch. Cropped to match.');
        end
    end

    % HDR-VDP-3 expects either 2D luminance or 3ch RGB; we use luminance.
    res = hdrvdp3('quality', gen_L, ref_L, color_encoding, ppd, opt_cell);

    names(i)  = string(stem);
    gpaths(i) = string(gp);
    rpaths(i) = string(rp);
    P_det(i)  = res.P_det;
    C_max(i)  = res.C_max;
    Q(i)      = res.Q;
    Q_JOD(i)  = res.Q_JOD;

    if ~quiet
        fprintf('  P_det=%6.4f  Q=%6.3f  Q_JOD=%6.3f\n', res.P_det, res.Q, res.Q_JOD);
    end

    if ~isempty(maps_dir)
        try
            Pm = max(min(res.P_map,1),0);
            imwrite(Pm, fullfile(maps_dir, sprintf('%s_Pmap.png', stem)));

            Cm = res.C_map;
            ub = prctile(Cm(:), 98);
            if ub <= 0, ub = max(Cm(:)); end
            if ub <= 0, ub = 1; end
            imwrite(mat2gray(Cm, [0 ub]), fullfile(maps_dir, sprintf('%s_Cmap.png', stem)));
        catch me
            if ~quiet
                warning('Failed to save maps for %s: %s', stem, me.message);
            end
        end
    end
end

results = table(names, gpaths, rpaths, P_det, C_max, Q, Q_JOD, ...
    'VariableNames', {'name','gen_path','ref_path','P_det','C_max','Q','Q_JOD'});

Q_mean   = mean(Q,    'omitnan');
Q_std    = std(Q,     'omitnan');
QJ_mean  = mean(Q_JOD,'omitnan');
QJ_std   = std(Q_JOD, 'omitnan');

if ~quiet
    fprintf('\nSummary (over %d pairs):\n', numel(common));
    fprintf('  Q:      mean = %.6f   std = %.6f\n',  Q_mean,  Q_std);
    fprintf('  Q_JOD:  mean = %.6f   std = %.6f\n\n', QJ_mean, QJ_std);
end

if ~isempty(out_csv)
    writetable(results, out_csv);
    fid = fopen(out_csv, 'a');
    if fid ~= -1
        fprintf(fid, 'mean,,,,,%.6f,%.6f\n', Q_mean, QJ_mean);
        fprintf(fid, 'std,,,,,%.6f,%.6f\n',  Q_std,  QJ_std);
        fclose(fid);
    end
    if ~quiet
        fprintf('Wrote CSV: %s\n', out_csv);
    end
end
end

% ---------------- helpers ----------------

function img = read_hdr_any(path)
% Read .exr or .hdr into single HxWxC
[~,~,ext] = fileparts(path);
ext = lower(ext);
switch ext
    case '.exr'
        img = exrread(path);
        img = to_float_exr(img);
    case '.hdr'
        img = hdrread(path);     % MATLAB Radiance HDR
        img = single(img);
    otherwise
        error('Unsupported file extension: %s', ext);
end
end

function a = to_float_exr(a)
if isa(a,'uint8') || isa(a,'uint16')
    a = im2single(a);
elseif ~isa(a,'single')
    a = single(a);
end
if isstruct(a)
    f = fieldnames(a);
    lf = lower(string(f));
    if all(ismember(["r","g","b"], lf))
        a = cat(3, a.R, a.G, a.B);
    elseif any(lf=="y")
        a = a.Y;
    else
        vals = {};
        for k = 1:numel(f)
            if isnumeric(a.(f{k})), vals{end+1} = a.(f{k}); end %#ok<AGROW>
        end
        if isempty(vals), error('Unsupported EXR struct format.'); end
        a = cat(3, vals{:});
    end
    a = single(a);
elseif iscell(a)
    for k = 1:numel(a), a{k} = single(a{k}); end
    a = cat(3, a{:});
end
if ndims(a) == 3 && size(a,3) > 3
    a = a(:,:,1:3);
end
end

function L = rgb2luminance_709(I)
if ndims(I) == 2 || size(I,3) == 1
    L = single(I);
else
    L = 0.212656*I(:,:,1) + 0.715158*I(:,:,2) + 0.072186*I(:,:,3);
end
end

function L_scaled = scale_to_peak_percentile(I_or_Y, L_peak, p, eps_floor)
if nargin < 3 || isempty(p), p = 99.9; end
if nargin < 4 || isempty(eps_floor), eps_floor = 1e-6; end

Y = I_or_Y;
Yv = Y(isfinite(Y) & Y > 0);
if isempty(Yv), error('All luminance values are non-finite or <= 0.'); end
pval = prctile(Yv, p);
pval = max(pval, eps_floor);
k = L_peak / pval;
L_scaled = I_or_Y * k;
end

function [A2, B2] = crop_to_match(A, B)
% Center-crop larger to smaller
ha = size(A,1); wa = size(A,2);
hb = size(B,1); wb = size(B,2);
h = min(ha,hb); w = min(wa,wb);
A2 = crop_center(A, h, w);
B2 = crop_center(B, h, w);
end

function I = crop_center(I, crop_h, crop_w)
[h, w] = size(I);
y1 = floor((h - crop_h)/2) + 1;
x1 = floor((w - crop_w)/2) + 1;
I = I(y1:y1+crop_h-1, x1:x1+crop_w-1);
end
