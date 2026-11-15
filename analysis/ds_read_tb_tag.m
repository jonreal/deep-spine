function [step, value] = ds_read_tb_tag(T, model_id, tag)
% DS_READ_TB_TAG Extract sorted step/value vectors for a model+tag pair.
  arguments
    T table
    model_id (1,1) string
    tag (1,1) string
  end

  mask = strcmp(T.ModelID, model_id) & strcmp(T.Tag, tag);
  [step, idx] = sort(T.Step(mask));
  value = T.Value(mask);
  value = value(idx);
end
