from .crew import RadiologyCrew

def generate_report(image_path: str) -> str:
    """Generate radiology report for uploaded image"""
    crew = RadiologyCrew().crew()
    result = crew.kickoff(inputs={"image_path": image_path})
    return str(result).strip()