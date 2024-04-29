import torch.distributed as dist
import logging


def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    tensor.div_(world_size)
    return tensor

def log_args(log_file):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)

def params_count(transfer_type, model):
    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    freezed_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logging.info("Total Params:%.3fM"%(model_total_params/1000000))
    logging.info("Encoder Params:%.3fM"%(freezed_params/1000000))
    logging.info("Gradient Params:%.3fM"%(model_grad_params/1000000))
    logging.info("Tuned percent [Grad/Total]:%.3f percent"%(model_grad_params/model_total_params*100))
    
    if transfer_type=="full_finetuning_3D":
        logging.info("This is 3D full finetuning, do not need to count the parameters.")
    
    elif transfer_type=="scratch":
        logging.info("This is train from scratch.")

    elif transfer_type=="head":
        decoder_params = model_grad_params
        logging.info("Decoder Params:%.3fM"%(decoder_params/1000000))
        logging.info("This is Head-Tune, only need to finetuning Decoder.")

    elif transfer_type=="med_adapter":
        med_adapter = sum(p[1].numel() for p in model.named_parameters() if "med_adapter" in p[0])
        logging.info("MedAdapter Params:%.3fM"%(med_adapter/1000000))
        logging.info("Tuned percent[MedAdapter/Total]:%.3f percent"%(med_adapter/model_total_params*100))

    else:
        raise ValueError("transfer type '{}' is not supported".format(transfer_type))

def freeze_params(transfer_type, model):
    if transfer_type=="full_finetuning_3D":
        logging.info("This is 3D full finetuning, do not need freeze.")

    elif transfer_type=="scratch":
        logging.info("This is train from scratch, do not need freeze.")

    elif transfer_type == "head":
        for name, param in model.named_parameters():
            if "patch_embed" in name:
                param.requires_grad = False
            if ".layers." in name: 
                param.requires_grad = False

    elif transfer_type == "med_adapter":
        for name, param in model.named_parameters():
            if "patch_embed" in name:
                param.requires_grad = False
            if ".layers." in name: 
                param.requires_grad = False
            if "med_adapter" in name:
                param.requires_grad = True

    else:
        raise ValueError("transfer type '{}' is not supported".format(transfer_type))