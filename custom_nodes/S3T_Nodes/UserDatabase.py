import os
from dotenv import load_dotenv
from supabase import create_client, Client

class NetworkConnectionRequired(Exception):
    pass
# TODO Add an Model API pulling that rather env for HF api user
#      needs to login get the api then pull the model and this way i can controll who acces the model

class Usagereporter:
    """Usage Reporter """

    def __init__(self):
        load_dotenv()
        url: str = os.getenv("SUPABASE_URL")
        key: str = os.getenv("SUPABASE_PUBLISHABLE_KEY")
        self.supabase:Client = create_client(url, key)
        self.user = None
        self.MAIL_ID:str=os.getenv("SUPABASE_MAIL")
        self.PASS:str=os.getenv("SUPABASE_PASS")

        if not self.MAIL_ID or not self.PASS:
             raise RuntimeError("configure credentials")

             
    def CredentialLogin(self):
        """User Log_In """
        try:
            session=self.supabase.auth.sign_in_with_password({
            "email":self.MAIL_ID,"password":self.PASS})
            self.user = session.user
        except Exception as e:
                Issue=str(e)
                if "Errno -3" in Issue or "Temporary failure in name resolution" in Issue:
                         raise NetworkConnectionRequired("Internet Connection required Please Connect to WIFI")
                else:
                     raise RuntimeError(Issue)

    def CurrentUsage(self) -> float:
        """Get the Current usage from the Database """
        if not self.user :
            raise PermissionError ("Must be logged in user ")
        try: 
            Get=(
                self.supabase.table("Client_Usage")
                .select("Usage_Seconds")
                .eq("client_id",self.user.id)
                .maybe_single()
                .execute())
            if Get is None or Get.data is None:
                return 0
            return Get.data.get("Usage_Seconds")
        
        except Exception as e: 
            raise RuntimeError(f"Fetching User usage Failed Due to \n{e}")
        
    def UpdateUsage(self,Used:int):
        """<|Update Usage with Last Use|>"""
        if Used < 0 :
            raise ValueError("Duration is less than 0 ")
        Current=self.CurrentUsage()
        Update :float = Current + Used 
        try:
            response=(
                 self.supabase.table("Client_Usage")
                .upsert({"client_id":self.user.id,"Usage_Seconds":Update})
                .execute())
            return response
        except Exception as e: 
            raise RuntimeError(f"Updating User Usage Failed \n{e}")

    def HF_api(self)-> str:
         """Acces hf Api from the database """
         if not self.user :
            raise PermissionError ("Must be logged in user ")
         try: 
             Get=(
                 self.supabase.table("Client_Usage")
                 .select("Hf_api")
                 .eq("client_id",self.user.id)
                 .maybe_single()
                 .execute())
             if Get is None or Get.data is None:
                 raise ConnectionRefusedError("invalid hf token either not registerd or plan expired")
             return Get.data.get("Hf_api")
         # 
         
         except Exception as e: 
            raise RuntimeError(f"\n{e}")
        
