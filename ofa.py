from ofa_model.zeroShot import OFA_model

def execute(
        model_path=None,
        data_path=None,
        model=None,
        extend_path=None,
        model_extend_path=None,
        batch_size=16,
        max_flavors=3,
        num_beams=1
        ):
    if(model=="OFA"):
        OFA=OFA_model(model_path)
        return OFA.getCaption(data_path,batch_size,num_beams)
    if(model=="OFA-embedding"):
        OFA=OFA_model(model_path)
        res=OFA.getEmbeddings(data_path,extend_path,model_extend_path,batch_size,num_beams)
        del OFA
        return res

if __name__=='__main__':
    execute(
        model_path="./models/OFA-large-caption",
        data_path="./static",
        model="OFA",
        extend_path=None,
        model_extend_path=None,
        batch_size=16,
        max_flavors=3,
        num_beams=1
    )