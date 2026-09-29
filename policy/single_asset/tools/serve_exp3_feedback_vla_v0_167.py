"""Same local service/robot contract with the registered learned feedback head."""
import serve_exp3_recurrent_vla_v0_166 as service
import exp3_oft_feedback_adapter_v0_167 as feedback

service.adapter=feedback

if __name__=='__main__':service.main()
