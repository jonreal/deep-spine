function save_pdf(fig, file, res)
%SAVE_PDF  Export a figure as the manuscript PDF.
%   SAVE_PDF(fig, file) writes vector PDF (exportgraphics, tight crop).
%   SAVE_PDF(fig, file, res) writes a full-page raster PDF at RES ppi with
%   print -image, the page sized to the figure. Used for the sweep heatmaps:
%   exportgraphics raster PDFs crop the bottom in batch mode.
  arguments
    fig
    file (1,:) char
    res = []
  end
  drawnow;
  if isempty(res)
    exportgraphics(fig, file, 'ContentType', 'vector');
  else
    fig.Units = 'points';
    fig.PaperUnits = 'points';
    fig.PaperSize = fig.Position(3:4);
    fig.PaperPosition = [0 0 fig.Position(3:4)];
    print(fig, file, '-dpdf', '-image', sprintf('-r%d', res));
  end
  fprintf('wrote %s\n', file);
end
