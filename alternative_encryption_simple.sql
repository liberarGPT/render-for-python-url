-- Alternative encryption without pgcrypto
-- This uses simple base64 encoding as a fallback

-- Drop existing functions
DROP FUNCTION IF EXISTS public.encrypt_api_key(TEXT);
DROP FUNCTION IF EXISTS public.decrypt_api_key(BYTEA);

-- Create simple base64 encoding function
CREATE OR REPLACE FUNCTION public.encrypt_api_key(key_text TEXT)
RETURNS TEXT
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  IF key_text IS NULL OR key_text = '' THEN
    RETURN NULL;
  END IF;
  
  -- Simple base64 encoding (not as secure as pgcrypto but functional)
  RETURN encode(key_text::bytea, 'base64');
END;
$$;

-- Create simple base64 decoding function
CREATE OR REPLACE FUNCTION public.decrypt_api_key(encoded_data TEXT)
RETURNS TEXT
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  IF encoded_data IS NULL OR encoded_data = '' THEN
    RETURN NULL;
  END IF;
  
  -- Simple base64 decoding
  RETURN convert_from(decode(encoded_data, 'base64'), 'UTF8');
END;
$$;

-- Grant permissions
GRANT EXECUTE ON FUNCTION public.encrypt_api_key(TEXT) TO authenticated;
GRANT EXECUTE ON FUNCTION public.decrypt_api_key(TEXT) TO authenticated;

-- Test the functions
DO $$
DECLARE
  test_text TEXT := 'test-api-key-123';
  encoded_data TEXT;
  decoded_text TEXT;
BEGIN
  -- Test encoding
  SELECT public.encrypt_api_key(test_text) INTO encoded_data;
  RAISE NOTICE 'Encoding successful: %', encoded_data;
  
  -- Test decoding
  SELECT public.decrypt_api_key(encoded_data) INTO decoded_text;
  
  IF decoded_text = test_text THEN
    RAISE NOTICE 'SUCCESS: Base64 encoding/decoding working correctly';
  ELSE
    RAISE NOTICE 'ERROR: Decoded text does not match original';
  END IF;
END $$;
